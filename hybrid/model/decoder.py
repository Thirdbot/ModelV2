"""Decoder for the grounded seismic VLM.

- `GroundedDecoder` — Qwen (4-bit) + frozen geology adapter + grounding adapter +
                      trainable fuse combiner (stacked-adapter fuse). The main
                      model uses `.decoder` + `.tokenizer`; facts enter as
                      digit-token embeddings and the LM copies the exact numbers.
"""
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from hybrid.data.config import (
    DECODER_MODEL_NAME,
    LOAD_DECODER_IN_4BIT,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    USE_LORA,
)


class GroundedDecoder(nn.Module):
    """Qwen (4-bit) + a stacked-adapter FUSE: geology + grounding + fuse LoRAs.

    The fuse is progressive LoRA stacking (a trainable combiner over frozen
    adapters): `set_stage` picks which adapters are active (additive) and which
    single one trains — geology is always frozen.
      stage 's2' : [geology, grounding] active, grounding trains (evidence-copy)
      stage 's3' : [geology, grounding, fuse] active, fuse trains (alignment)
    Exposes `.decoder` (the PEFT LM) and `.tokenizer`."""

    def __init__(self, decoder_model_name=DECODER_MODEL_NAME, use_lora=USE_LORA,
                 lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                 load_in_4bit=LOAD_DECODER_IN_4BIT, adapter_dir=None):
        super().__init__()
        self._decoder_4bit = bool(load_in_4bit and torch.cuda.is_available())
        # Load the tokenizer the geology adapter was trained with, so vocab lines up.
        self.tokenizer = AutoTokenizer.from_pretrained(adapter_dir or decoder_model_name)
        if self._decoder_4bit:
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                     bnb_4bit_use_double_quant=True,
                                     bnb_4bit_compute_dtype=torch.bfloat16)
            self.decoder = AutoModelForCausalLM.from_pretrained(
                decoder_model_name, quantization_config=bnb,
                torch_dtype=torch.bfloat16, device_map={"": 0})
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)
        # Grow the embedding table to the adapter's vocab if needed (never shrink).
        if len(self.tokenizer) > self.decoder.get_input_embeddings().weight.shape[0]:
            self.decoder.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if use_lora:
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
            )
            if self._decoder_4bit:
                # No gradient checkpointing: the narrator's sequences are short, and
                # checkpointing + inputs_embeds severs the grad path.
                self.decoder = prepare_model_for_kbit_training(
                    self.decoder, use_gradient_checkpointing=False)
            if adapter_dir is not None:
                # geology (frozen, from stage 1) + grounding + fuse LoRAs. Both
                # new adapters start at delta=0; set_stage controls active/trainable.
                self.decoder = PeftModel.from_pretrained(
                    self.decoder, adapter_dir, adapter_name="geology", is_trainable=False)
                self.decoder.add_adapter("grounding", lora_config)
                self.decoder.add_adapter("fuse", lora_config)
                self.set_stage("s2")
            else:
                self.decoder = get_peft_model(self.decoder, lora_config)

    def set_stage(self, stage):
        """Pick active (additive) adapters and the single trainable one; geology
        always frozen. 's2' trains grounding; 's3' trains the fuse combiner."""
        active = {"s2": ["geology", "grounding"],
                  "s3": ["geology", "grounding", "fuse"]}[stage]
        train_name = {"s2": "grounding", "s3": "fuse"}[stage]
        self.decoder.base_model.set_adapter(active)
        for n, p in self.decoder.named_parameters():
            if "lora_" in n:
                p.requires_grad_(train_name in n)

    def to(self, *args, **kwargs):
        """Leave the 4-bit decoder where bitsandbytes pinned it; move other children."""
        if getattr(self, "_decoder_4bit", False):
            for name, child in self.named_children():
                if name != "decoder":
                    child.to(*args, **kwargs)
            return self
        return super().to(*args, **kwargs)
