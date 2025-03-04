import numpy as np
import os
from transformers import (
    AutoTokenizer,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    AutoModelForCausalLM,
    TrainerCallback,
)
from fastchat.model import get_conversation_template

from llm_exploration.bandits.llm_bandit import LLMBandit
from llm_exploration.utils.stats_utils import get_cumulative_average
from llm_exploration.bandits.bandit_utils import get_empirical_regret
from llm_exploration.utils.data_utils import write_json
from llm_exploration.inference.huggingface_inference_engine import (
    HuggingFaceLLMInferenceEngine,
)


class BanditTrainerCallback(TrainerCallback):
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        bandit_env: LLMBandit,
        n_steps: int,
        model_name: str,
        savedir: str,
        num_trials: int,
    ):
        self.tokenizer = tokenizer
        self.bandit_env = bandit_env
        self.n_steps = n_steps
        self.savedir = savedir
        self.model_name = model_name
        self.num_trials = num_trials

        self.step_count = 0

    def eval_loop_on_bandit(self, inference_engine: HuggingFaceLLMInferenceEngine):
        conv = get_conversation_template("gpt-4")
        if self.bandit_env.should_include_system_prompt_in_conversation():
            conv.set_system_message(self.bandit_env.get_system_prompt())

        probability_list = np.random.uniform(
            low=0.0,
            high=1.0,
            size=len(self.bandit_env.get_probability_list()),
        ).tolist()

        self.bandit_env.reset(
            probability_list=probability_list,
        )

        for time_step in range(self.bandit_env.T):
            user_prompt = self.bandit_env.get_user_prompt(time_step=time_step)
            conv.append_message(
                role="user",
                message=user_prompt,
            )

            model_response = inference_engine.generate(
                conv=conv.to_openai_api_messages(),
                max_n_tokens=512,
                temperature=0.0,
                top_p=1.0,
            )

            conv.append_message(
                role="assistant",
                message=model_response,
            )

            self.bandit_env.step_text_action(
                text_action=model_response,
            )

        all_rewards = self.bandit_env.get_rewards()
        cumulative_average_rewards = get_cumulative_average(
            arr=all_rewards,
        )
        empirical_regret = get_empirical_regret(
            bandit_env=self.bandit_env,
        )
        history = self.bandit_env.generate_history(use_text_actions=True)
        bandit_probability_list = self.bandit_env.get_probability_list()
        text_actions = self.bandit_env.get_all_text_actions()
        num_invalid_actions = self.bandit_env.get_num_invalid_actions()

        return {
            "conversation": history,
            "all_rewards": all_rewards,
            "cumulative_average_rewards": cumulative_average_rewards,
            "empirical_regret": empirical_regret,
            "probability_list": bandit_probability_list,
            "text_actions": text_actions,
            "num_invalid_actions": num_invalid_actions,
        }

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> None:
        self.step_count += 1

        if self.step_count % self.n_steps == 0:
            model: AutoModelForCausalLM = kwargs["model"]
            tokenizer_padding_side = self.tokenizer.padding_side

            inference_engine = HuggingFaceLLMInferenceEngine(
                model_name=self.model_name,
                model=model,
                tokenizer=self.tokenizer,
            )

            all_histories = []
            all_empirical_regrets = []
            for _ in range(self.num_trials):
                record = self.eval_loop_on_bandit(inference_engine=inference_engine)
                all_histories.append(record)
                all_empirical_regrets.append(record["empirical_regret"])

            self.tokenizer.padding_side = tokenizer_padding_side

            mean_empirical_regret = np.mean(all_empirical_regrets, axis=0).tolist()

            save_path = os.path.join(self.savedir, f"step_{self.step_count}.json")

            write_json(
                data={
                    "num_trials": self.num_trials,
                    "mean_empirical_regret": mean_empirical_regret,
                    "records": all_histories,
                },
                fname=save_path,
            )
