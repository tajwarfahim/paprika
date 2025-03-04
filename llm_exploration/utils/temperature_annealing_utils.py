from abc import ABC, abstractmethod
from typing import Optional


class TemperatureScheduler(ABC):
    @abstractmethod
    def step(self) -> None:
        pass

    @abstractmethod
    def get_temperature(self) -> float:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ConstTemperatureScheduler(TemperatureScheduler):
    """
    Used to return a constant temperature scheduler
    """

    def __init__(
        self,
        temp: float,
        num_total_steps: int,
    ):
        self.temp = temp
        self.curr_step = 0
        self.num_total_steps = num_total_steps

    def step(self) -> None:
        self.curr_step += 1

    def get_temperature(self) -> float:
        return self.temp

    def reset(self) -> None:
        self.curr_step = 0


class LinearTemperatureScheduler(TemperatureScheduler):
    """
    Used for a linear increase/decrease in temperature
    """

    def __init__(
        self,
        init_temp: float,
        final_temp: float,
        num_total_steps: int,
    ):
        self.init_temp = init_temp
        self.final_temp = final_temp

        self.curr_step = 0
        self.num_total_steps = num_total_steps

    def step(self) -> None:
        self.curr_step += 1

    def get_temperature(self) -> float:
        difference = self.final_temp - self.init_temp
        curr_temp = self.init_temp + (float(difference) * self.curr_step) / (
            self.num_total_steps - 1
        )
        return curr_temp

    def reset(self) -> None:
        self.curr_step = 0


def get_min_p_from_temperature(
    temperature: float,
    temperature_threshold: float,
    min_p_choice: float,
) -> Optional[float]:
    """
    Given temperature for sampling from an LLM, this chooses
    the min_p base probability hyperparameter.
    For temperature > 0.5, we use min_p = 0.1
    For temperature <= 0.5, we do not use min_p (set min_p = None)

    Input:
        temperature (float):
            The temperature parameter for LLM generation.
            Higher temperature means more variability in the generations,
            whereas lower temperature means more deterministic answers.

            See the following link for documentation:
            https://platform.openai.com/docs/api-reference/introduction

        temperature_threshold (float):
            We only use min_p sampling for temperature above this threshold

        min_p_choice (float):
            Value of p_base for min_p sampling, if used

    Output:
        min_p (float or None):
            Alternative way of sampling/decoding from the LLM
            Please see this paper for more details: https://arxiv.org/abs/2407.01082
            Values must be between 0 and 1, with typically 0.01-0.2 being preferred.

            NOTE 1: Typically useful when one wants to sample at high temperatures,
            e.g., temperature > 1

            NOTE 2: Also currenlty only supported in huggingface, and not in OpenAI

            NOTE 3: If None, then min_p is not used
    """
    if temperature >= temperature_threshold:
        return min_p_choice
    else:
        return None
