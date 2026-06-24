from data import TemplateDataset,qea_process
from VLM import VLM
if __name__ == '__main__':
    from unsloth import FastVisionModel
    from data import LangCollator
    from trl import SFTConfig,SFTTrainer
    from transformers import EarlyStoppingCallback
    from pathlib import Path

    root= Path(__file__).parent

    stage1_model = (root / "trained-image-evidences").as_posix()


    ie = TemplateDataset("thirdExec/synthetic-seismic-vlm",map_fn=qea_process)

    model,processor = VLM(stage1_model).load_unsloth_vlm(use_gradient_checkpointing="unsloth",load_in_4bit=True)

    processor.image_processor.do_image_splitting = False # natively the processor is split image, but we are passing it different size anyway

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Wait 3 eval evaluations without improvement
        early_stopping_threshold=0.01  # Minimum improvement required
    )

    training_args = SFTConfig(
        output_dir="./trained-question-evidences-answer",
        max_length=512,
        save_steps=1000,
        save_strategy='epoch',
        fp16=False,
        bf16=False,
        dataset_kwargs={
            "add_special_tokens": False,  # Let the chat template handle structural tokens
            "skip_prepare_dataset": True,  # already prepared for collate
        },
        remove_unused_columns=False,
        num_train_epochs=1000,
        eval_strategy='epoch',
        learning_rate=2e-5,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False ,

    )

    collator = LangCollator(processor=processor) # language only

    # load state from stage 1
    model = model.merge_and_unload()

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=8,
        lora_alpha=8,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=ie.train_dataset,
        processing_class=processor.tokenizer,
        data_collator=collator,
        eval_dataset=ie.eval_dataset,
        callbacks=[early_stopping_callback]

    )
    trainer.train(resume_from_checkpoint = False)
    model.save_pretrained("./trained-question-evidences-answer")
    processor.tokenizer.save_pretrained("./trained-question-evidences-answer")
    processor.save_pretrained("./trained-question-evidences-answer")