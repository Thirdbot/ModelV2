import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, EncoderDecoderConfig, PreTrainedModel


"""
Grounding VQA model for seismic interpretation.

The model receives one or more full seismic images plus a question. It generates
answer text with structured region tags. Each generated <SEG> token is used as
the grounding anchor for the bbox and mask heads.
"""


class VLM(PreTrainedModel):
    config_class = EncoderDecoderConfig

    def __init__(
        self,
        config,
        tokenizer,
        encoder_name_or_path=None,
        decoder_name_or_path=None,
        max_detection_slots=10,
        max_image_slots=64,
        image_size=224,
        text_loss_weight=1.0,
        bbox_loss_weight=1.0,
        mask_loss_weight=1.0,
        verbose=False,
    ):
        super().__init__(config)
        self.width = image_size
        self.height = image_size
        self.tokenizer = tokenizer
        self.max_detection_slots = max_detection_slots
        self.max_image_slots = max_image_slots
        self.text_loss_weight = text_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self._last_image_infos = None

        if decoder_name_or_path is None:
            self.decoder = AutoModelForCausalLM.from_config(config.decoder)
        else:
            self.decoder = AutoModelForCausalLM.from_pretrained(decoder_name_or_path)

        if encoder_name_or_path is None:
            self.encoder = AutoModel.from_config(config.encoder)
        else:
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)

        self.config.encoder = self.encoder.config
        self.config.decoder = self.decoder.config

        encoder_hidden_size = self.encoder.config.hidden_size
        decoder_hidden_size = self.decoder.config.hidden_size
        self.bridge_adapter = nn.Sequential(
            nn.LayerNorm(encoder_hidden_size),
            nn.Linear(encoder_hidden_size, decoder_hidden_size),
            nn.GELU(),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
        )
        self.image_geometry_mlp = nn.Sequential(
            nn.Linear(4, decoder_hidden_size),
            nn.GELU(),
            nn.Linear(decoder_hidden_size, decoder_hidden_size),
        )
        self.image_index_embed = nn.Embedding(max_image_slots, decoder_hidden_size)
        self.object_queries = nn.Embedding(max_detection_slots, decoder_hidden_size)
        self.bbox_head = nn.Sequential(
            nn.LayerNorm(decoder_hidden_size),
            nn.Linear(decoder_hidden_size, 4),
        )
        mask_hidden_size = min(256, decoder_hidden_size)
        self.mask_patch_head = nn.Linear(decoder_hidden_size, mask_hidden_size)
        self.mask_object_head = nn.Linear(decoder_hidden_size, mask_hidden_size)
        self.mask_decoder = nn.Sequential(
            nn.Conv2d(mask_hidden_size, mask_hidden_size, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden_size, mask_hidden_size // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mask_hidden_size // 2, 1, kernel_size=1),
        )

        if verbose:
            print("initializing VLM")
            print("creating bridge for enc-dec", self.bridge_adapter)
            print(self.tokenizer.chat_template)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_name_or_path,
        decoder_name_or_path,
        tokenizer=None,
        max_detection_slots=10,
        max_image_slots=64,
        image_size=224,
        text_loss_weight=1.0,
        bbox_loss_weight=1.0,
        mask_loss_weight=1.0,
        verbose=False,
    ):
        encoder_config = AutoConfig.from_pretrained(encoder_name_or_path)
        decoder_config = AutoConfig.from_pretrained(decoder_name_or_path)
        config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=encoder_config,
            decoder_config=decoder_config,
        )
        if tokenizer is None:
            tokenizer = AutoProcessor.from_pretrained(decoder_name_or_path)
        return cls(
            config=config,
            tokenizer=tokenizer,
            encoder_name_or_path=encoder_name_or_path,
            decoder_name_or_path=decoder_name_or_path,
            max_detection_slots=max_detection_slots,
            max_image_slots=max_image_slots,
            image_size=image_size,
            text_loss_weight=text_loss_weight,
            bbox_loss_weight=bbox_loss_weight,
            mask_loss_weight=mask_loss_weight,
            verbose=verbose,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        return self.decoder.resize_token_embeddings(*args, **kwargs)

    def _normalize_full_image_batch(self, pixel_values):
        if not isinstance(pixel_values, list):
            return None
        if not pixel_values:
            return []
        if isinstance(pixel_values[0], torch.Tensor):
            return [pixel_values]
        return pixel_values

    def _resize_pad_image(self, image):
        _, orig_h, orig_w = image.shape
        scale = min(float(self.width) / float(orig_w), float(self.height) / float(orig_h))
        resized_w = max(1, round(orig_w * scale))
        resized_h = max(1, round(orig_h * scale))
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(resized_h, resized_w),
            mode="bilinear",
            align_corners=False,
        )[0]
        padded = image.new_zeros(image.shape[0], self.height, self.width)
        padded[:, :resized_h, :resized_w] = resized
        return padded, {
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
            "resized_h": int(resized_h),
            "resized_w": int(resized_w),
            "scale": float(scale),
            "pad_x": 0,
            "pad_y": 0,
        }

    def _resize_pad_mask(self, mask, info, device):
        mask = mask.to(device=device, dtype=torch.float32)
        if mask.ndim == 3:
            mask = mask[0]
        resized = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0),
            size=(int(info["resized_h"]), int(info["resized_w"])),
            mode="nearest",
        )[0, 0]
        padded = mask.new_zeros(self.height, self.width)
        padded[: int(info["resized_h"]), : int(info["resized_w"])] = resized
        return (padded > 0).float()

    def build_full_image_prefix_inputs(self, pixel_values, input_ids, attention_mask=None, regions=None):
        full_batch = self._normalize_full_image_batch(pixel_values)
        if full_batch is None:
            return None

        device = input_ids.device
        text_dtype = self.decoder.get_input_embeddings().weight.dtype
        all_images = []
        image_owners = []
        per_example_infos = [[] for _ in full_batch]

        for batch_idx, images in enumerate(full_batch):
            for image_index, image in enumerate(images):
                image = image.to(device=device, dtype=torch.float32)
                encoded_image, info = self._resize_pad_image(image)
                info.update({"batch_index": batch_idx, "image_index": image_index})
                all_images.append(encoded_image)
                image_owners.append(info)
                per_example_infos[batch_idx].append(info)

        if not all_images:
            raise ValueError("No images were provided to the VLM.")

        encoded_images = torch.stack(all_images).to(device=device)
        vision_out = self.encoder(pixel_values=encoded_images).last_hidden_state
        image_embeds = self.bridge_adapter(vision_out).to(dtype=text_dtype)

        orig_w = torch.tensor([info["orig_w"] for info in image_owners], dtype=torch.float32, device=device)
        orig_h = torch.tensor([info["orig_h"] for info in image_owners], dtype=torch.float32, device=device)
        resized_w = torch.tensor([info["resized_w"] for info in image_owners], dtype=torch.float32, device=device)
        resized_h = torch.tensor([info["resized_h"] for info in image_owners], dtype=torch.float32, device=device)
        image_indices = torch.tensor([info["image_index"] for info in image_owners], dtype=torch.long, device=device)
        image_embeds = self.add_image_metadata_embeds(
            image_embeds=image_embeds,
            orig_w=orig_w,
            orig_h=orig_h,
            resized_w=resized_w,
            resized_h=resized_h,
            image_indices=image_indices,
        )

        per_example_embeds = []
        cursor = 0
        for infos in per_example_infos:
            count = len(infos)
            embeds = image_embeds[cursor:cursor + count].reshape(-1, image_embeds.shape[-1])
            for offset, info in enumerate(infos):
                info["embeds"] = image_embeds[cursor + offset]
            per_example_embeds.append(embeds)
            cursor += count

        max_visual_len = max(embeds.shape[0] for embeds in per_example_embeds)
        padded_embeds = image_embeds.new_zeros(len(per_example_embeds), max_visual_len, image_embeds.shape[-1])
        image_attention = torch.zeros(len(per_example_embeds), max_visual_len, dtype=torch.long, device=device)
        for batch_idx, embeds in enumerate(per_example_embeds):
            padded_embeds[batch_idx, : embeds.shape[0]] = embeds
            image_attention[batch_idx, : embeds.shape[0]] = 1

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([padded_embeds, text_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([image_attention.to(dtype=attention_mask.dtype), attention_mask], dim=1)
        self._last_image_infos = per_example_infos
        return vision_out, padded_embeds, inputs_embeds, attention_mask

    def build_prefix_inputs(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
        **kwargs,
    ):
        full_prefix = self.build_full_image_prefix_inputs(pixel_values, input_ids, attention_mask)
        if full_prefix is not None:
            return full_prefix

        vision_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        image_embeds = self.bridge_adapter(vision_out).to(dtype=text_embeds.dtype)
        image_embeds = self.add_image_metadata_embeds(
            image_embeds=image_embeds,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
        )
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        batch_size, image_seq_len, _ = image_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        image_attention = torch.ones(
            batch_size,
            image_seq_len,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        return vision_out, image_embeds, inputs_embeds, attention_mask

    def add_image_metadata_embeds(
        self,
        image_embeds,
        orig_w=None,
        orig_h=None,
        resized_w=None,
        resized_h=None,
        image_indices=None,
    ):
        batch_size = image_embeds.shape[0]
        device = image_embeds.device
        dtype = image_embeds.dtype

        if orig_w is None:
            orig_w = torch.full((batch_size,), float(self.width), device=device, dtype=dtype)
        else:
            orig_w = orig_w.to(device=device, dtype=dtype).clamp_min(1.0)
        if orig_h is None:
            orig_h = torch.full((batch_size,), float(self.height), device=device, dtype=dtype)
        else:
            orig_h = orig_h.to(device=device, dtype=dtype).clamp_min(1.0)
        if resized_w is None:
            resized_w = torch.full((batch_size,), float(self.width), device=device, dtype=dtype)
        else:
            resized_w = resized_w.to(device=device, dtype=dtype).clamp_min(1.0)
        if resized_h is None:
            resized_h = torch.full((batch_size,), float(self.height), device=device, dtype=dtype)
        else:
            resized_h = resized_h.to(device=device, dtype=dtype).clamp_min(1.0)

        geometry = torch.stack(
            [
                orig_w / float(self.width),
                orig_h / float(self.height),
                resized_w / float(self.width),
                resized_h / float(self.height),
            ],
            dim=-1,
        )
        geometry_embed = self.image_geometry_mlp(geometry.float()).to(dtype=dtype)

        if image_indices is None:
            image_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            image_indices = image_indices.to(device=device, dtype=torch.long)
        image_indices = image_indices.clamp(min=0, max=self.max_image_slots - 1)
        image_embed = self.image_index_embed(image_indices).to(dtype=dtype)

        return image_embeds + (geometry_embed + image_embed).unsqueeze(1)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
        **generate_kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if pixel_values is None:
            raise ValueError("pixel_values are required")

        _, _, inputs_embeds, attention_mask = self.build_prefix_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
        )
        return self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

    @torch.no_grad()
    def predict_grounding(self, pixel_values, num_objects=None, orig_w=None, orig_h=None, image_indices=None):
        full_batch = self._normalize_full_image_batch(pixel_values)
        if full_batch is not None:
            batch_size = len(full_batch)
            device = next(self.parameters()).device
            dummy_input_ids = torch.zeros(batch_size, 1, dtype=torch.long, device=device)
            _, image_embeds, _, _ = self.build_full_image_prefix_inputs(
                pixel_values=full_batch,
                input_ids=dummy_input_ids,
                attention_mask=torch.ones_like(dummy_input_ids),
            )
            return self.predict_full_image_grounding(image_embeds=image_embeds, num_objects=num_objects)

        vision_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        text_dtype = self.decoder.get_input_embeddings().weight.dtype
        image_embeds = self.bridge_adapter(vision_out).to(dtype=text_dtype)
        image_embeds = self.add_image_metadata_embeds(
            image_embeds=image_embeds,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
        )
        return self.forward_grounding(image_embeds=image_embeds, num_objects=num_objects)

    @torch.no_grad()
    def predict_grounding_from_text(self, pixel_values, input_ids, attention_mask=None, grounding_image_indices=None):
        full_batch = self._normalize_full_image_batch(pixel_values)
        if full_batch is None:
            raise ValueError("predict_grounding_from_text expects full-image pixel_values as a list.")

        _, image_embeds, inputs_embeds, attention_mask = self.build_full_image_prefix_inputs(
            pixel_values=full_batch,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        object_embeds = self.object_embeds_from_seg_tokens(
            input_ids=input_ids,
            decoder_hidden_states=decoder_outputs.hidden_states[-1],
            image_seq_len=image_embeds.shape[1],
            image_embeds=image_embeds,
            target_slots=self.max_detection_slots,
        )
        num_objects = self.count_seg_slots(input_ids)
        return self.predict_full_image_grounding(
            image_embeds=image_embeds,
            num_objects=num_objects,
            object_embeds=object_embeds,
            grounding_image_indices=grounding_image_indices,
        )

    def predict_full_image_grounding(self, image_embeds, num_objects=None, object_embeds=None, grounding_image_indices=None):
        if self._last_image_infos is None:
            raise ValueError("Full-image grounding prediction requires image metadata.")

        image_embeds = image_embeds.float()
        batch_size = image_embeds.shape[0]
        if num_objects is None:
            num_objects = self.max_detection_slots
        num_objects = min(num_objects, self.max_detection_slots)
        if object_embeds is None:
            cls_embed = image_embeds[:, 0]
            queries = self.object_queries.weight[:num_objects].unsqueeze(0).expand(batch_size, -1, -1)
            object_embeds = queries + cls_embed.unsqueeze(1)
        else:
            object_embeds = object_embeds[:, :num_objects].float()

        bbox_unit_preds = torch.sigmoid(self.bbox_head(object_embeds))
        bbox_preds = bbox_unit_preds * float(self.width)
        mask_logits = image_embeds.new_zeros(batch_size, num_objects, self.height, self.width)
        image_meta = []

        for batch_idx in range(batch_size):
            infos = self._last_image_infos[batch_idx]
            image_meta.append([])
            if not infos:
                continue
            for obj_idx in range(num_objects):
                image_index = None
                if grounding_image_indices is not None and obj_idx < len(grounding_image_indices[batch_idx]):
                    image_index = int(grounding_image_indices[batch_idx][obj_idx])
                info = self.select_image_info(infos, image_index=image_index, fallback_slot=obj_idx)
                mask_logits[batch_idx, obj_idx] = self.decode_mask_from_patches(
                    object_embed=object_embeds[batch_idx, obj_idx],
                    image_embed=info["embeds"],
                )
                image_meta[batch_idx].append({key: value for key, value in info.items() if key != "embeds"})

        return {
            "bbox_preds": bbox_preds,
            "mask_logits": mask_logits,
            "image_meta": image_meta,
            "grounding_loss": None,
            "bbox_loss": None,
            "mask_loss": None,
            "mask_bce_loss": None,
            "mask_dice_loss": None,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pixel_values=None,
        bbox_targets=None,
        mask_targets=None,
        class_ids=None,
        object_mask=None,
        masks=None,
        regions=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if pixel_values is None:
            raise ValueError("pixel_values are required")

        vision_out, image_embeds, inputs_embeds, attention_mask = self.build_prefix_inputs(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
        )

        if labels is not None:
            image_labels = torch.full(
                (image_embeds.shape[0], image_embeds.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([image_labels, labels], dim=1)

        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
        )

        loss = decoder_outputs.loss
        weighted_text_loss = None if loss is None else loss * self.text_loss_weight
        output = {
            "loss": weighted_text_loss,
            "logits": decoder_outputs.logits,
            "text_loss": loss,
            "vision_last_hidden_state": vision_out,
        }

        if regions is not None and masks is not None:
            seg_object_embeds = self.object_embeds_from_seg_tokens(
                input_ids=input_ids,
                decoder_hidden_states=decoder_outputs.hidden_states[-1],
                image_seq_len=image_embeds.shape[1],
                image_embeds=image_embeds,
                target_slots=self.max_detection_slots,
            )
            detection_output = self.forward_full_image_grounding(
                image_embeds=image_embeds,
                masks=masks,
                regions=regions,
                object_embeds=seg_object_embeds,
            )
            output.update(detection_output)
            output["loss"] = detection_output["grounding_loss"] if weighted_text_loss is None else weighted_text_loss + detection_output["grounding_loss"]
        elif bbox_targets is not None or mask_targets is not None:
            seg_object_embeds = self.object_embeds_from_seg_tokens(
                input_ids=input_ids,
                decoder_hidden_states=decoder_outputs.hidden_states[-1],
                image_seq_len=image_embeds.shape[1],
                image_embeds=image_embeds,
                target_slots=self._target_slot_count(bbox_targets, mask_targets),
            )
            detection_output = self.forward_grounding(
                image_embeds=image_embeds,
                bbox_targets=bbox_targets,
                mask_targets=mask_targets,
                object_mask=object_mask,
                object_embeds=seg_object_embeds,
            )
            output.update(detection_output)
            output["loss"] = detection_output["grounding_loss"] if weighted_text_loss is None else weighted_text_loss + detection_output["grounding_loss"]

        return output

    def bbox_to_encoder_frame(self, bbox, info, device):
        x1, y1, x2, y2 = [float(value) for value in bbox]
        scale = float(info["scale"])
        bbox_tensor = torch.tensor(
            [x1 * scale, y1 * scale, x2 * scale, y2 * scale],
            dtype=torch.float32,
            device=device,
        )
        return bbox_tensor.clamp(min=0.0, max=float(self.width))

    def decode_mask_from_patches(self, object_embed, image_embed):
        patch_embeds = self.mask_patch_head(image_embed[1:].float())
        object_embed = self.mask_object_head(object_embed.float())
        patch_embeds = torch.nan_to_num(patch_embeds, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        object_embed = torch.nan_to_num(object_embed, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        patch_count = patch_embeds.shape[0]
        grid_size = int(patch_count ** 0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Expected square patch grid, got {patch_count} patches")
        patch_grid = patch_embeds.reshape(grid_size, grid_size, -1)
        conditioned = patch_grid * object_embed.view(1, 1, -1)
        conditioned = conditioned.permute(2, 0, 1).unsqueeze(0)
        slot_mask_logits = self.mask_decoder(conditioned)[0, 0]
        mask_logits = F.interpolate(
            slot_mask_logits.unsqueeze(0).unsqueeze(0),
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        return torch.nan_to_num(mask_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)

    def mask_criterion(self, logits, targets):
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        targets = torch.nan_to_num(targets.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(min=0.0, max=1.0)

        flat_targets = targets.flatten(start_dim=1)
        positives = flat_targets.sum(dim=1)
        valid_objects = positives > 0
        if not valid_objects.any():
            zero = logits.sum() * 0.0
            return zero, zero, zero

        logits = logits[valid_objects]
        targets = targets[valid_objects]
        positives = positives[valid_objects]
        negatives = targets[0].numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=20.0)

        bce_per_pixel = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        weights = torch.ones_like(targets)
        weight_shape = [pos_weight.shape[0]] + [1] * (targets.ndim - 1)
        weights = torch.where(targets > 0, pos_weight.view(*weight_shape), weights)
        bce_loss = (bce_per_pixel * weights).flatten(start_dim=1).mean(dim=1).mean()

        probs = torch.sigmoid(logits)
        smooth = 1.0
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
        dice_loss = dice_loss.mean()

        mask_loss = bce_loss + dice_loss
        mask_loss = torch.nan_to_num(mask_loss, nan=0.0, posinf=0.0, neginf=0.0)
        bce_loss = torch.nan_to_num(bce_loss, nan=0.0, posinf=0.0, neginf=0.0)
        dice_loss = torch.nan_to_num(dice_loss, nan=0.0, posinf=0.0, neginf=0.0)
        return mask_loss, bce_loss, dice_loss

    def forward_full_image_grounding(self, image_embeds, masks, regions, object_embeds=None):
        if self._last_image_infos is None:
            raise ValueError("Full-image grounding requires image metadata.")

        image_embeds = image_embeds.float()
        batch_size = len(regions)
        num_objects = self.max_detection_slots
        device = image_embeds.device

        if object_embeds is None:
            cls_embed = image_embeds[:, 0]
            queries = self.object_queries.weight[:num_objects].unsqueeze(0).expand(batch_size, -1, -1)
            object_embeds = queries + cls_embed.unsqueeze(1)
        else:
            object_embeds = object_embeds[:, :num_objects].float()

        bbox_unit_preds = torch.sigmoid(self.bbox_head(object_embeds))
        bbox_preds = bbox_unit_preds * float(self.width)
        mask_logits = image_embeds.new_zeros(batch_size, num_objects, self.height, self.width)
        bbox_targets = image_embeds.new_zeros(batch_size, num_objects, 4)
        mask_targets = image_embeds.new_zeros(batch_size, num_objects, self.height, self.width)
        object_mask = torch.zeros(batch_size, num_objects, dtype=torch.bool, device=device)

        for batch_idx, batch_regions in enumerate(regions):
            infos = self._last_image_infos[batch_idx]
            if not infos:
                continue
            for obj_idx, region in enumerate(batch_regions[:num_objects]):
                image_index = int(region.get("image_index", 0))
                info = next((candidate for candidate in infos if int(candidate["image_index"]) == image_index), None)
                if info is None:
                    continue
                mask_index = int(region.get("mask_index", region.get("region_index", 0)))
                if mask_index >= len(masks[batch_idx]):
                    continue

                bbox_targets[batch_idx, obj_idx] = self.bbox_to_encoder_frame(region["bbox"], info, device)
                mask_targets[batch_idx, obj_idx] = self._resize_pad_mask(masks[batch_idx][mask_index], info, device)
                mask_logits[batch_idx, obj_idx] = self.decode_mask_from_patches(
                    object_embed=object_embeds[batch_idx, obj_idx],
                    image_embed=info["embeds"],
                )
                object_mask[batch_idx, obj_idx] = True

        grounding_loss = image_embeds.new_tensor(0.0)
        bbox_loss = None
        mask_loss = None
        mask_bce_loss = None
        mask_dice_loss = None
        if object_mask.any():
            bbox_unit_targets = bbox_targets / float(self.width)
            bbox_loss = F.smooth_l1_loss(bbox_unit_preds[object_mask], bbox_unit_targets[object_mask], beta=0.05)
            bbox_loss = torch.nan_to_num(bbox_loss, nan=0.0, posinf=0.0, neginf=0.0)
            mask_loss, mask_bce_loss, mask_dice_loss = self.mask_criterion(
                mask_logits[object_mask],
                mask_targets[object_mask],
            )
            grounding_loss = self.bbox_loss_weight * bbox_loss + self.mask_loss_weight * mask_loss
            grounding_loss = torch.nan_to_num(grounding_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "bbox_preds": bbox_preds,
            "mask_logits": mask_logits,
            "bbox_targets": bbox_targets,
            "mask_targets": mask_targets,
            "object_mask": object_mask,
            "grounding_loss": grounding_loss,
            "bbox_loss": bbox_loss,
            "mask_loss": mask_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
        }

    def select_image_info(self, infos, image_index=None, fallback_slot=0):
        if image_index is not None:
            for info in infos:
                if int(info["image_index"]) == int(image_index):
                    return info
        return infos[min(fallback_slot, len(infos) - 1)]

    def _target_slot_count(self, bbox_targets=None, mask_targets=None):
        if bbox_targets is not None:
            return min(bbox_targets.shape[1], self.max_detection_slots)
        if mask_targets is not None:
            return min(mask_targets.shape[1], self.max_detection_slots)
        return self.max_detection_slots

    def object_embeds_from_seg_tokens(
        self,
        input_ids,
        decoder_hidden_states,
        image_seq_len,
        image_embeds,
        target_slots,
    ):
        if input_ids is None or self.tokenizer is None:
            return None

        seg_token_id = self.tokenizer.convert_tokens_to_ids("<SEG>")
        if seg_token_id is None or seg_token_id < 0:
            return None

        batch_size = input_ids.shape[0]
        cls_embed = image_embeds[:, 0].float()
        queries = self.object_queries.weight[:target_slots].unsqueeze(0).expand(batch_size, -1, -1)
        object_embeds = queries + cls_embed.unsqueeze(1)

        for batch_idx in range(batch_size):
            seg_positions = (input_ids[batch_idx] == seg_token_id).nonzero(as_tuple=False).flatten()
            if seg_positions.numel() == 0:
                continue
            seg_positions = seg_positions[:target_slots] + image_seq_len
            object_embeds[batch_idx, : seg_positions.numel()] = decoder_hidden_states[
                batch_idx,
                seg_positions,
            ].float()

        return object_embeds

    def count_seg_slots(self, input_ids):
        if input_ids is None or self.tokenizer is None:
            return self.max_detection_slots
        seg_token_id = self.tokenizer.convert_tokens_to_ids("<SEG>")
        if seg_token_id is None or seg_token_id < 0:
            return self.max_detection_slots
        count = int((input_ids == seg_token_id).sum(dim=1).max().item())
        return min(count, self.max_detection_slots)

    def forward_grounding(
        self,
        image_embeds,
        bbox_targets=None,
        mask_targets=None,
        object_mask=None,
        object_embeds=None,
        num_objects=None,
    ):
        image_embeds = image_embeds.float()
        batch_size = image_embeds.shape[0]
        cls_embed = image_embeds[:, 0]
        patch_embeds = image_embeds[:, 1:]

        if num_objects is None:
            num_objects = self._target_slot_count(bbox_targets, mask_targets)
        num_objects = min(num_objects, self.max_detection_slots)

        if object_embeds is None:
            queries = self.object_queries.weight[:num_objects].unsqueeze(0).expand(batch_size, -1, -1)
            object_embeds = queries + cls_embed.unsqueeze(1)
        else:
            object_embeds = object_embeds[:, :num_objects].float()

        bbox_unit_preds = torch.sigmoid(self.bbox_head(object_embeds))
        bbox_preds = bbox_unit_preds * float(self.width)

        patch_embeds = self.mask_patch_head(patch_embeds)
        object_mask_embeds = self.mask_object_head(object_embeds)
        patch_embeds = torch.nan_to_num(patch_embeds, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        object_mask_embeds = torch.nan_to_num(object_mask_embeds, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)
        patch_count = patch_embeds.shape[1]
        grid_size = int(patch_count ** 0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Expected square patch grid, got {patch_count} patches")

        patch_grid = patch_embeds.reshape(batch_size, 1, grid_size, grid_size, -1)
        object_grid = object_mask_embeds.reshape(batch_size, num_objects, 1, 1, -1)
        conditioned = (patch_grid * object_grid).permute(0, 1, 4, 2, 3)
        conditioned = conditioned.reshape(batch_size * num_objects, conditioned.shape[2], grid_size, grid_size)
        mask_logits = self.mask_decoder(conditioned).reshape(batch_size, num_objects, grid_size, grid_size)
        mask_logits = F.interpolate(
            mask_logits,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        mask_logits = torch.nan_to_num(mask_logits, nan=0.0, posinf=20.0, neginf=-20.0).clamp(min=-20.0, max=20.0)

        grounding_loss = image_embeds.new_tensor(0.0)
        bbox_loss = None
        mask_loss = None
        mask_bce_loss = None
        mask_dice_loss = None

        if object_mask is None:
            object_mask = torch.ones(batch_size, num_objects, dtype=torch.bool, device=image_embeds.device)
        else:
            object_mask = object_mask[:, :num_objects].to(image_embeds.device)

        if bbox_targets is not None and object_mask.any():
            bbox_targets = bbox_targets[:, :num_objects].to(image_embeds.device).float()
            bbox_unit_targets = bbox_targets / float(self.width)
            bbox_loss = F.smooth_l1_loss(bbox_unit_preds[object_mask], bbox_unit_targets[object_mask], beta=0.05)
            bbox_loss = torch.nan_to_num(bbox_loss, nan=0.0, posinf=0.0, neginf=0.0)
            grounding_loss = grounding_loss + self.bbox_loss_weight * bbox_loss

        if mask_targets is not None and object_mask.any():
            mask_targets = mask_targets[:, :num_objects].to(image_embeds.device).float()
            mask_loss, mask_bce_loss, mask_dice_loss = self.mask_criterion(
                mask_logits[object_mask],
                mask_targets[object_mask],
            )
            grounding_loss = grounding_loss + self.mask_loss_weight * mask_loss
            grounding_loss = torch.nan_to_num(grounding_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "bbox_preds": bbox_preds,
            "mask_logits": mask_logits,
            "grounding_loss": grounding_loss,
            "bbox_loss": bbox_loss,
            "mask_loss": mask_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
        }


if __name__ == "__main__":
    encoder = "NorskRegnesentralSTI/NCS-v1-2d-base"
    decoder = "HuggingFaceTB/SmolLM-360M-Instruct"

    tokenizer = AutoProcessor.from_pretrained(decoder)
    model = VLM.from_encoder_decoder_pretrained(encoder, decoder, tokenizer)
