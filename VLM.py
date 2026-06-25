from unsloth import FastVisionModel
from pathlib import Path
from transformers import AutoProcessor

class VLM:
    def __init__(self,model_path):
        self.model_path = model_path
        self.SPECIAL_TOKENS = [
                              "<SEG>",
        ]

    def load_unsloth_vlm(self, **args):
          model, processor = FastVisionModel.from_pretrained(self.model_path, **args)

          if Path(self.model_path, "tokenizer.json").exists():
              processor = AutoProcessor.from_pretrained(self.model_path)

          vocab = processor.tokenizer.get_vocab()
          missing_tokens = [
              token for token in self.SPECIAL_TOKENS
              if token not in vocab
          ]

          if missing_tokens:
              num_added = processor.tokenizer.add_special_tokens(
                  {"additional_special_tokens": missing_tokens}
              )
              print(f"Added {num_added} new special tokens: {missing_tokens}")
              model.resize_token_embeddings(len(processor.tokenizer))
          else:
              print("Skip adding special tokens, new tokens already added")

          return model, processor
