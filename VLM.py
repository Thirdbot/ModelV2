from unsloth import FastVisionModel

class VLM:
    def __init__(self,model_path):
        self.model_path = model_path
        self.SPECIAL_TOKENS = [
                              "<region>",
                              "</region>",
                              "<object>",
                              "</object>",
                              "<class_id>",
                              "</class_id>",
                              "<color>",
                              "</color>",
                              "<evidence>",
                              "</evidence>",
                              "<think>",
                              "</think>",
                              "<answer>",
                              "</answer>",
                              "<SEG>",
        ]

    def load_unsloth_vlm(self,**args):
        model,processor = FastVisionModel.from_pretrained(self.model_path,**args)

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

        return model,processor
