import torch
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl

from data import clean_messages, ensure_image_list


class PrintOnPredictTextCallback(TrainerCallback):
    def __init__(self,trainer,processor=None,tokenizer=None,num_samples=3,max_new_tokens=120):
        self.trainer = trainer
        self.processor = processor
        self.tokenizer = tokenizer if tokenizer is not None else processor.tokenizer
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        eval_dataset = self.trainer.eval_dataset
        if hasattr(eval_dataset, "select"):
            sample_indices = list(range(min(self.num_samples, len(eval_dataset))))
            samples = eval_dataset.select(sample_indices)
        else:
            # Fallback for standard indexing
            samples = [eval_dataset[i] for i in range(min(self.num_samples, len(eval_dataset)))]

        print(f"\n=== Evaluation Samples at Step {state.global_step} ===")

        # 2. Put model in evaluation mode
        model = self.trainer.model
        model.eval()

        with torch.no_grad():
            for i, sample in enumerate(samples):
                messages = clean_messages(sample["message"])
                prompt_messages = messages[:1]

                prompt_text = self.tokenizer.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                target_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

                processor_kwargs = {
                    "text": [prompt_text],
                    "padding": True,
                    "return_tensors": "pt",
                }
                if "i" in sample:
                    processor_kwargs["images"] = [ensure_image_list(sample["i"])]

                inputs = self.processor(**processor_kwargs).to(model.device)

                generated_outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

                gen_tokens = generated_outputs[0][inputs["input_ids"].shape[1]:]
                generated_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

                print(f"\nSample {i + 1}:")
                print(f"  [PROMPT]:    {prompt_text.strip()}")
                print(f"  [TARGET]:    {target_text.strip()}")
                print(f"  [GENERATED]: {generated_text.strip()}")
                print("-" * 40)

        # Return model back to training state
        model.train()
