from transformers import Trainer


def safe_save_model_for_hf_trainer(
    trainer: Trainer,
    output_dir: str,
) -> None:
    """
    Collects the state dict and dump to disk.

    Input:
        trainer (transformers.Trainer):
            The huggingface transformers object

        output_dir (str):
            Directory where the model weights/state dict needs to be saved.

    Output:
        None
    """
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)
