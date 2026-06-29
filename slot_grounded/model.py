import torch
import torch.nn as nn
from torchvision.transforms.functional import pil_to_tensor
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.data import SPECIAL_TOKENS
from utils.region_encoder import NcsEncoder, RegionEncoder


def xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(dim=-1)
    return torch.stack(
        [
            (x1 + x2) * 0.5,
            (y1 + y2) * 0.5,
            (x2 - x1).clamp_min(0),
            (y2 - y1).clamp_min(0),
        ],
        dim=-1,
    )


def cxcywh_to_xyxy(values):
    pred = values.sigmoid()
    cx, cy, w, h = pred.unbind(dim=-1)
    x1 = (cx - 0.5 * w).clamp(0, 1)
    y1 = (cy - 0.5 * h).clamp(0, 1)
    x2 = (cx + 0.5 * w).clamp(0, 1)
    y2 = (cy + 0.5 * h).clamp(0, 1)
    return torch.stack([x1, y1, x2, y2], dim=-1)


class VisualPrefixBridge(nn.Module):
    def __init__(self, vision_dim=768, decoder_dim=960, bbox_dim=7, num_classes=6):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(vision_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.region_proj = nn.Sequential(
            nn.Linear(vision_dim, decoder_dim),
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

    def forward(self, global_features, region_features=None, region_bbox_norm=None, class_ids=None):
        dtype = self.global_proj[0].weight.dtype
        global_tokens = self.global_proj(global_features.to(dtype=dtype))
        global_tokens = global_tokens + self.type_embed.weight[0]

        if region_features is None or region_bbox_norm is None or class_ids is None:
            return global_tokens

        region_tokens = self.region_proj(region_features.to(dtype=dtype))
        region_tokens = region_tokens + self.type_embed.weight[1]

        bbox_tokens = self.bbox_proj(region_bbox_norm.to(dtype=dtype))
        bbox_tokens = bbox_tokens + self.class_embed(class_ids)
        bbox_tokens = bbox_tokens + self.type_embed.weight[2]

        return torch.cat([global_tokens, region_tokens, bbox_tokens], dim=0)


class SegMaskDecoder(nn.Module):
    def __init__(self, decoder_dim, vision_dim=768, base_channels=128):
        super().__init__()
        self.query_proj = nn.Sequential(
            nn.Linear(decoder_dim, base_channels),
            nn.GELU(),
        )
        self.spatial_proj = nn.Conv2d(vision_dim, base_channels, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels, 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=2, stride=2),
        )

    def forward(self, seg_hidden, spatial_features):
        dtype = self.query_proj[0].weight.dtype
        seg_hidden = seg_hidden.to(dtype=dtype)
        spatial_features = spatial_features.to(dtype=dtype)
        query = self.query_proj(seg_hidden).unsqueeze(-1).unsqueeze(-1)
        x = self.spatial_proj(spatial_features) + query
        return self.decoder(x)


class SlotGroundedVLM(nn.Module):
    def __init__(self, decoder_model_name="HuggingFaceTB/SmolLM-360M-Instruct"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name)
        missing_tokens = [
            token for token in SPECIAL_TOKENS
            if token not in self.tokenizer.get_vocab()
        ]
        if missing_tokens:
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": missing_tokens}
            )
            self.decoder.resize_token_embeddings(len(self.tokenizer))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        decoder_dim = self.decoder.get_input_embeddings().embedding_dim
        self.ncs_encoder = NcsEncoder()
        self.roi = RegionEncoder(output_size=224, spatial_scale=1, sampling_ratio=2)
        self.tile_pos = nn.Sequential(
            nn.Linear(4, 256),
            nn.GELU(),
            nn.Linear(256, 768),
        )
        self.bridge = VisualPrefixBridge(decoder_dim=decoder_dim)
        self.object_head = nn.Linear(decoder_dim, 7)
        self.bbox_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 4),
        )
        self.center_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )
        self.numeric_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 1),
        )
        self.mask_decoder = SegMaskDecoder(decoder_dim=decoder_dim)

    def encode_tiles(self, tiles, device):
        features = []
        positions = []
        for tile in tiles:
            image = tile["image"]
            tensor = pil_to_tensor(image.convert("RGB")).float().unsqueeze(0).to(device) / 255.0
            feature = self.ncs_encoder(tensor)
            features.append(feature)
            positions.append(tile["bbox_norm"])

        features = torch.cat(features, dim=0)
        positions = torch.tensor(
            positions,
            dtype=self.tile_pos[0].weight.dtype,
            device=features.device,
        )
        return features.to(dtype=self.tile_pos[-1].weight.dtype) + self.tile_pos(positions)

    def encode_one_image(self, pixel_values, tiles, bbox, class_ids, height, width):
        device = next(self.parameters()).device
        pixel_values = pixel_values.to(device)

        global_features = self.encode_tiles(tiles, device)
        crops, bbox_norm = self.roi(pixel_values, bbox, height, width)
        region_features, spatial_features = self.ncs_encoder(crops, return_spatial=True)
        class_ids = torch.as_tensor(class_ids, dtype=torch.long, device=device)
        if class_ids.ndim == 0:
            class_ids = class_ids.unsqueeze(0)

        return {
            "global_features": global_features,
            "region_features": region_features,
            "region_spatial_features": spatial_features,
            "region_bbox_norm": bbox_norm,
            "class_ids": class_ids,
        }

    def encode_images(self, pixel_values, tiles, boxes, class_ids, heights, widths):
        outputs = []
        for idx, pixel_value in enumerate(pixel_values):
            outputs.append(
                self.encode_one_image(
                    pixel_values=pixel_value,
                    tiles=tiles[idx],
                    bbox=boxes[idx],
                    class_ids=class_ids[idx],
                    height=heights[idx],
                    width=widths[idx],
                )
            )
        return outputs

    def build_visual_tokens(self, visual_output):
        return self.bridge(
            global_features=visual_output["global_features"],
        )

    def group_visual_tokens(self, visual_outputs, row_image_counts):
        tokens = [self.build_visual_tokens(output) for output in visual_outputs]
        grouped = []
        cursor = 0
        for count in row_image_counts:
            grouped.append(torch.cat(tokens[cursor:cursor + count], dim=0))
            cursor += count
        return grouped

    def forward(
        self,
        pixel_values,
        tiles,
        boxes,
        class_ids,
        heights,
        widths,
        input_ids,
        attention_mask,
        labels=None,
        row_image_counts=None,
    ):
        visual_outputs = self.encode_images(
            pixel_values=pixel_values,
            tiles=tiles,
            boxes=boxes,
            class_ids=class_ids,
            heights=heights,
            widths=widths,
        )

        device = input_ids.device
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        visual_tokens = self.group_visual_tokens(visual_outputs, row_image_counts)
        max_visual_len = max(token.shape[0] for token in visual_tokens)
        padded_visual_tokens = []
        visual_attention = []
        for token in visual_tokens:
            token = token.to(device=device, dtype=text_embeds.dtype)
            pad_len = max_visual_len - token.shape[0]
            if pad_len:
                token = torch.cat(
                    [
                        token,
                        torch.zeros(
                            pad_len,
                            token.shape[-1],
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
        visual_attention = torch.stack(visual_attention, dim=0)
        inputs_embeds = torch.cat([visual_tokens, text_embeds], dim=1)
        full_attention = torch.cat([visual_attention, attention_mask], dim=1)

        if labels is not None:
            visual_labels = torch.full(
                (labels.shape[0], max_visual_len),
                -100,
                device=labels.device,
                dtype=labels.dtype,
            )
            labels = torch.cat([visual_labels, labels], dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )
        return {
            "decoder_outputs": decoder_outputs,
            "visual_outputs": visual_outputs,
            "visual_len": max_visual_len,
            "loss": decoder_outputs.loss,
        }

    def slot_bbox_to_xyxy(self, hidden):
        hidden = hidden.to(dtype=self.bbox_head[0].weight.dtype)
        return cxcywh_to_xyxy(self.bbox_head(hidden))

    def slot_center_to_xy(self, hidden):
        hidden = hidden.to(dtype=self.center_head[0].weight.dtype)
        return self.center_head(hidden).sigmoid()
