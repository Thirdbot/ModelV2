from train_common import TrainConfig, train_from_config


TRAINING_TYPE = "text_mask_output"

CONFIG = TrainConfig(
    run_name="train_100",
    save_root="outputs",
    max_steps=-1,
    num_train_epochs=100,
    max_train_samples=None,
    max_eval_samples=None,
    max_length=256,
    save_only_model=False,
    use_wandb=True,
)


train_from_config(CONFIG, training_type=TRAINING_TYPE)
