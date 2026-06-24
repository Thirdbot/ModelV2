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


        num_added = processor.tokenizer.add_special_tokens(
                    {"additional_special_tokens": self.SPECIAL_TOKENS}
        )
        if num_added > 0:
            print("Added {} new tokens".format(num_added))
            model.resize_token_embeddings(len(processor.tokenizer))

        return model,processor
