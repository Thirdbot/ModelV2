from VLM import VLM
from callback import PrintOnPredictTextCallback
from data import TemplateDataset,ie_process

if __name__ == "__main__":
    from unsloth import FastVisionModel
    from data import VisionCollator
    from trl import SFTConfig,SFTTrainer
    from transformers import EarlyStoppingCallback

    ie = TemplateDataset("thirdExec/synthetic-seismic-vlm",map_fn=ie_process)
    print("dataset size: ",len(ie.temped_dataset))
    print("example: \n\t",ie.temped_dataset[0:2])

    model,processor = VLM("HuggingFaceTB/SmolVLM-500M-Instruct").load_unsloth_vlm(use_gradient_checkpointing="unsloth",load_in_4bit=True)

    processor.image_processor.do_image_splitting = False # natively the processor is split image, but we are passing it different size anyway

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Wait 3 eval evaluations without improvement
        early_stopping_threshold=0.01  # Minimum improvement required
    )

    training_args = SFTConfig(
        output_dir="../trained-image-evidences",
        max_length=1024,
        save_steps=1000,
        save_total_limit=2,
        save_strategy='epoch',
        fp16=False,
        bf16=False,
        dataset_kwargs={
            "add_special_tokens": False,  # Let the chat template handle structural tokens
            "skip_prepare_dataset": True,  # already prepared for collate
        },
        remove_unused_columns=False,
        num_train_epochs=21,
        eval_strategy='epoch',
        learning_rate=2e-5,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        greater_is_better=False ,

    )

    collator = VisionCollator(processor=processor)
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
    log_callback = PrintOnPredictTextCallback(trainer=trainer, processor=processor, num_samples=1)
    trainer.add_callback(log_callback)
    trainer.train(resume_from_checkpoint = False)
    model.save_pretrained_merged("./trained-image-evidences/fw",processor.tokenizer,save_method='merged_16bit')
    processor.tokenizer.save_pretrained("./trained-image-evidences/fw")
    processor.save_pretrained("./trained-image-evidences/fw")
