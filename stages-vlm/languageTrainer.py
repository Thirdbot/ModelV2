from trl import SFTConfig, SFTTrainer


training_args = SFTConfig(
    output_dir="./output",
    max_length=1024,
    save_steps=1000,
    save_total_limit=2,
    save_strategy='epoch',
    fp16=False,
    bf16=False,
    dataset_kwargs={
        "add_special_tokens": False,    # Let the chat template handle structural tokens
        "skip_prepare_dataset": True, # already prepared for collate
    },
    remove_unused_columns=False,
    num_train_epochs=100,
    eval_strategy='epoch',
    learning_rate=2e-5,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False
)


if __name__ == '__main__':
    from unsloth import FastVisionModel
    from data import TemplateDataset, VisionCollator,preprocess_fn
    from transformers import EarlyStoppingCallback
    from pathlib import Path
    from VLM import VLM

    root= Path(__file__).parent

    stage2_model = (root / "trained-question-evidences-answer/fw").as_posix()

    # Quantized for smaller size vlm
    model, processor = VLM(stage2_model).load_unsloth_vlm(use_gradient_checkpointing="unsloth",
                                                                                   load_in_4bit=True) # for native model, in future use from_config for custom model

    # not using this in the future ,but it is proof of concept for changing training procedure
    processor.image_processor.do_image_splitting = False # natively the processor is split image, but we are passing it different size anyway

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=10,  # Wait 3 eval evaluations without improvement
        early_stopping_threshold=0.01  # Minimum improvement required
    )

    # for ncs model this could be just resize the input of ncs rather than tiling it
    dataset = TemplateDataset("thirdExec/synthetic-seismic-vlm",map_fn=preprocess_fn)
    print("dataset size: ",len(dataset.temped_dataset))
    print("example: \n\t",dataset.temped_dataset[0:2])

    collator = VisionCollator(processor=processor)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
        r=8,
        lora_alpha=8,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset.train_dataset,
        processing_class=processor.tokenizer,
        data_collator=collator,
        eval_dataset=dataset.eval_dataset,
        callbacks=[early_stopping_callback]
    )
    trainer.train(resume_from_checkpoint = False)
    model = model.merge_and_unload()
    model.save_pretrained("./output/fw")
    processor.tokenizer.save_pretrained("./output/fw")
    processor.save_pretrained("./output/fw")