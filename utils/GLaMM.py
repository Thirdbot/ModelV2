import torch.nn as nn

from utils.global_region_encoder import DualEncoder
from utils.lang_decoder import Decoder


class SeismicGLaMM(nn.Module):
    def __init__(self, decoder_model_name="HuggingFaceTB/SmolLM-360M-Instruct"):
        super().__init__()
        self.dual_encoder = DualEncoder(is_train=True)
        self.lang_decoder = Decoder(model_name=decoder_model_name)

    def forward(
        self,
        pixel_values,
        tiles,
        bbox=None,
        class_ids=None,
        H=None,
        W=None,
        input_ids=None,
        attention_mask=None,
        labels=None,
        row_image_counts=None,
    ):
        dual_outputs = self.dual_encoder(
            pixel_values=pixel_values,
            tiles=tiles,
            bbox=bbox,
            class_ids=class_ids,
            H=H,
            W=W,
        )

        if input_ids is None or attention_mask is None:
            return {
                "dual_outputs": dual_outputs,
                "decoder_outputs": None,
            }

        decoder_outputs = self.lang_decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            dual_encoder_outputs=dual_outputs,
            row_image_counts=row_image_counts,
        )

        return {
            "dual_outputs": dual_outputs,
            "decoder_outputs": decoder_outputs,
            "loss": decoder_outputs.loss,
        }

    @property
    def tokenizer(self):
        return self.lang_decoder.tokenizer
