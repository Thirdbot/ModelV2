import torch
import torch.nn as nn
from transformers import AutoTokenizer,AutoModelForCausalLM
from utils.data import SPECIAL_TOKENS


class VLBridge(nn.Module):
    def __init__(
            self,
            encoder_dim=768,
            decoder_dim=960,
            num_classes=6,
            bbox_dim=7,
    ):
        super().__init__()

        self.global_proj = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        self.region_proj = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        self.bbox_proj = nn.Sequential(
            nn.Linear(bbox_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )

        self.class_embed = nn.Embedding(num_classes + 1, decoder_dim)

        self.type_embed = nn.Embedding(3, decoder_dim)
        # 0 = global, 1 = region, 2 = bbox/class

    def forward(
            self,
            global_tiles,
            region_features,
            region_bbox_norm,
            region_class_ids=None,
    ):
        # global_tiles: [T, 768]
        # region_features: [R, 768]
        # region_bbox_norm: [R, bbox_dim]

        global_tokens = self.global_proj(global_tiles)
        global_tokens = global_tokens + self.type_embed.weight[0]

        region_tokens = self.region_proj(region_features)
        region_tokens = region_tokens + self.type_embed.weight[1]

        bbox_tokens = self.bbox_proj(region_bbox_norm)

        if region_class_ids is not None:
            bbox_tokens = bbox_tokens + self.class_embed(region_class_ids)

        bbox_tokens = bbox_tokens + self.type_embed.weight[2]

        visual_tokens = torch.cat(
            [global_tokens, region_tokens, bbox_tokens],
            dim=0,
        )

        return visual_tokens

class Decoder(nn.Module):
    def __init__(self, model_name="HuggingFaceTB/SmolLM-360M-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        missing_tokens = [
            token for token in SPECIAL_TOKENS
            if token not in self.tokenizer.get_vocab()
        ]
        if missing_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": missing_tokens}
            )
            self.model.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        decoder_dim = self.model.get_input_embeddings().embedding_dim
        self.vl_bridge = VLBridge(decoder_dim=decoder_dim)

    def _build_visual_tokens(self, dual_encoder_output, device):
        global_tiles = dual_encoder_output["global_tiles"].to(device)
        region_feature = dual_encoder_output["region_feature"].to(device)
        region_bbox_norm = dual_encoder_output["region_bbox_norm"].to(device)
        region_class_ids = dual_encoder_output.get("region_class_ids")
        if region_class_ids is not None:
            region_class_ids = region_class_ids.to(device)
        return self.vl_bridge(
            global_tiles=global_tiles,
            region_features=region_feature,
            region_bbox_norm=region_bbox_norm,
            region_class_ids=region_class_ids,
        )

    def forward(self,input_ids,attention_mask,dual_encoder_outputs,labels=None):
        device = input_ids.device
        text_embeds = self.model.get_input_embeddings()(input_ids)
        model_dtype = text_embeds.dtype

        if isinstance(dual_encoder_outputs, dict):
            dual_encoder_outputs = [dual_encoder_outputs]

        visual_tokens = [
            self._build_visual_tokens(output, device)
            for output in dual_encoder_outputs
        ]
        max_visual_len = max(token.shape[0] for token in visual_tokens)
        visual_dim = text_embeds.shape[-1]

        padded_visual_tokens = []
        visual_attention = []
        for token in visual_tokens:
            pad_len = max_visual_len - token.shape[0]
            if pad_len > 0:
                token = torch.cat(
                    [
                        token,
                        torch.zeros(
                            pad_len,
                            visual_dim,
                            device=device,
                            dtype=token.dtype,
                        ),
                    ],
                    dim=0,
                )
            padded_visual_tokens.append(token)
            visual_attention.append(
                torch.cat(
                    [
                        torch.ones(max_visual_len - pad_len, device=device, dtype=attention_mask.dtype),
                        torch.zeros(pad_len, device=device, dtype=attention_mask.dtype),
                    ],
                    dim=0,
                )
            )

        visual_tokens = torch.stack(padded_visual_tokens, dim=0)
        visual_tokens = visual_tokens.to(dtype=model_dtype)
        visual_attention = torch.stack(visual_attention, dim=0)

        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        attention_mask = torch.cat([visual_attention, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (labels.shape[0], max_visual_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
