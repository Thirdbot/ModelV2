from train_common import TrainConfig, train_from_config


TRAINING_TYPE = "text_mask_output"

CONFIG = TrainConfig(
    run_name="train_100",
    save_root="outputs",
    max_steps=-1,
    num_train_epochs=100,
    max_objects=4,
    max_train_samples=None,
    max_eval_samples=None,
    max_length=512,
    mask_loss_weight=8.0,
    text_loss_weight=0.2,
    learning_rate=1e-5,
    max_grad_norm=1.0,
    freeze_encoder=False,
    fp16=False,
    bf16=False,
    save_only_model=False,
    use_wandb=True,
)


train_from_config(CONFIG, training_type=TRAINING_TYPE)
