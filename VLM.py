import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedModel, EncoderDecoderConfig, AutoConfig, AutoProcessor, AutoModelForCausalLM

"""
This is the part where model construction happens

1. vision encoder must load
2. it must accept various 2d image size and images
3. it must connect vision prefix with additional tokens hidden layers
4. text decoder must load
5. it must load additional tokens as specials tokens
6. it must be able to follow format and given bbox
7. if there is mask decoder connect to it, it must output masking with same
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
        max_internal_tiles=64,
        internal_tile_stride=112,
        verbose=False,
    ):
        super().__init__(config)
        self.width, self.height = (224, 224) # by the Vision Encoder
        self.tokenizer = tokenizer
        self.max_detection_slots = max_detection_slots
        self.max_image_slots = max_image_slots
        self.max_internal_tiles = max_internal_tiles
        self.internal_tile_stride = internal_tile_stride
        self._last_tile_infos = None

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
        self.tile_position_mlp = nn.Sequential(
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
        self.mask_patch_head = nn.Linear(decoder_hidden_size, decoder_hidden_size)
        if verbose:
            print('initializing VLM')
            print('creating bridge for enc-dec',self.bridge_adapter) # passing features from vision to decoder as embeddings
            print(self.tokenizer.chat_template)

    @classmethod
    def from_encoder_decoder_pretrained(
        cls,
        encoder_name_or_path,
        decoder_name_or_path,
        tokenizer=None,
        max_detection_slots=10,
        max_image_slots=64,
        max_internal_tiles=64,
        internal_tile_stride=112,
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
            max_internal_tiles=max_internal_tiles,
            internal_tile_stride=internal_tile_stride,
            verbose=verbose,
        )

    def resize_token_embeddings(self, *args, **kwargs):
        return self.decoder.resize_token_embeddings(*args, **kwargs)

    def _tile_starts(self, length, tile_size, stride):
        if length <= tile_size:
            return [0]
        starts = list(range(0, length - tile_size + 1, stride))
        last = length - tile_size
        if starts[-1] != last:
            starts.append(last)
        return starts

    def _pad_to_tile(self, image):
        _, height, width = image.shape
        pad_h = max(self.height - height, 0)
        pad_w = max(self.width - width, 0)
        if pad_h or pad_w:
            image = F.pad(image, (0, pad_w, 0, pad_h))
        return image, {"orig_h": height, "orig_w": width, "pad_h": pad_h, "pad_w": pad_w}

    def _normalize_full_image_batch(self, pixel_values):
        if not isinstance(pixel_values, list):
            return None
        if not pixel_values:
            return []
        if isinstance(pixel_values[0], torch.Tensor):
            return [pixel_values]
        return pixel_values

    def build_full_image_prefix_inputs(self, pixel_values, input_ids, attention_mask=None, regions=None):
        full_batch = self._normalize_full_image_batch(pixel_values)
        if full_batch is None:
            return None

        device = input_ids.device
        text_dtype = self.decoder.get_input_embeddings().weight.dtype
        all_tiles = []
        tile_owners = []
        per_example_infos = [[] for _ in full_batch]

        for batch_idx, images in enumerate(full_batch):
            candidates = []
            for image_index, image in enumerate(images):
                image = image.to(device=device, dtype=torch.float32)
                image, pad_meta = self._pad_to_tile(image)
                _, padded_h, padded_w = image.shape
                xs = self._tile_starts(padded_w, self.width, self.internal_tile_stride)
                ys = self._tile_starts(padded_h, self.height, self.internal_tile_stride)
                for y0 in ys:
                    for x0 in xs:
                        score = self._tile_region_score(
                            x0=x0,
                            y0=y0,
                            regions=regions[batch_idx] if regions is not None else None,
                            image_index=image_index,
                        )
                        tile = image[:, y0:y0 + self.height, x0:x0 + self.width]
                        candidates.append((
                            score,
                            tile,
                            {
                                "batch_index": batch_idx,
                                "image_index": image_index,
                                "x0": x0,
                                "y0": y0,
                                **pad_meta,
                            },
                        ))

            candidates.sort(key=lambda item: item[0], reverse=True)
            for _, tile, info in candidates[:self.max_internal_tiles]:
                all_tiles.append(tile)
                tile_owners.append(info)
                per_example_infos[batch_idx].append(info)

        if not all_tiles:
            raise ValueError("No internal tiles were produced from full-image pixel_values.")

        tiles = torch.stack(all_tiles).to(device=device)
        vision_out = self.encoder(pixel_values=tiles).last_hidden_state
        image_embeds = self.bridge_adapter(vision_out).to(dtype=text_dtype)
        tile_x = torch.tensor([info["x0"] for info in tile_owners], dtype=torch.float32, device=device)
        tile_y = torch.tensor([info["y0"] for info in tile_owners], dtype=torch.float32, device=device)
        orig_w = torch.tensor([info["orig_w"] for info in tile_owners], dtype=torch.float32, device=device)
        orig_h = torch.tensor([info["orig_h"] for info in tile_owners], dtype=torch.float32, device=device)
        image_indices = torch.tensor([info["image_index"] for info in tile_owners], dtype=torch.long, device=device)
        image_embeds = self.add_tile_position_embeds(
            image_embeds=image_embeds,
            tile_x=tile_x,
            tile_y=tile_y,
            orig_w=orig_w,
            orig_h=orig_h,
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
            padded_embeds[batch_idx, :embeds.shape[0]] = embeds
            image_attention[batch_idx, :embeds.shape[0]] = 1

        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([padded_embeds, text_embeds], dim=1)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([image_attention.to(dtype=attention_mask.dtype), attention_mask], dim=1)
        self._last_tile_infos = per_example_infos
        return vision_out, padded_embeds, inputs_embeds, attention_mask

    def _tile_region_score(self, x0, y0, regions, image_index):
        if regions is None:
            return 0
        score = 0
        for region in regions:
            if int(region.get("image_index", 0)) != int(image_index):
                continue
            local_bbox = self._intersect_bbox(region["bbox"], x0, y0)
            if local_bbox is None:
                continue
            score += max(local_bbox[2] - local_bbox[0], 0) * max(local_bbox[3] - local_bbox[1], 0)
        return score

    def build_prefix_inputs(
        self,
        pixel_values,
        input_ids,
        attention_mask=None,
        tile_x=None,
        tile_y=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
        regions=None,
    ):
        full_prefix = self.build_full_image_prefix_inputs(pixel_values, input_ids, attention_mask, regions=regions)
        if full_prefix is not None:
            return full_prefix

        vision_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        text_embeds = self.decoder.get_input_embeddings()(input_ids)
        image_embeds = self.bridge_adapter(vision_out).to(dtype=text_embeds.dtype)
        image_embeds = self.add_tile_position_embeds(
            image_embeds=image_embeds,
            tile_x=tile_x,
            tile_y=tile_y,
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

    def add_tile_position_embeds(
        self,
        image_embeds,
        tile_x=None,
        tile_y=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
    ):
        batch_size = image_embeds.shape[0]
        device = image_embeds.device
        dtype = image_embeds.dtype

        if tile_x is None:
            tile_x = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            tile_x = tile_x.to(device=device, dtype=dtype)
        if tile_y is None:
            tile_y = torch.zeros(batch_size, device=device, dtype=dtype)
        else:
            tile_y = tile_y.to(device=device, dtype=dtype)
        if orig_w is None:
            orig_w = torch.full((batch_size,), float(self.width), device=device, dtype=dtype)
        else:
            orig_w = orig_w.to(device=device, dtype=dtype).clamp_min(1.0)
        if orig_h is None:
            orig_h = torch.full((batch_size,), float(self.height), device=device, dtype=dtype)
        else:
            orig_h = orig_h.to(device=device, dtype=dtype).clamp_min(1.0)

        tile_features = torch.stack(
            [
                tile_x / orig_w,
                tile_y / orig_h,
                orig_w / float(self.width),
                orig_h / float(self.height),
            ],
            dim=-1,
        )
        position_embed = self.tile_position_mlp(tile_features.float()).to(dtype=dtype)

        if image_indices is None:
            image_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            image_indices = image_indices.to(device=device, dtype=torch.long)
        image_indices = image_indices.clamp(min=0, max=self.max_image_slots - 1)
        image_embed = self.image_index_embed(image_indices).to(dtype=dtype)

        return image_embeds + (position_embed + image_embed).unsqueeze(1)

    @torch.no_grad()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        tile_x=None,
        tile_y=None,
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
            tile_x=tile_x,
            tile_y=tile_y,
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
    def predict_grounding(
        self,
        pixel_values,
        num_objects=None,
        tile_x=None,
        tile_y=None,
        orig_w=None,
        orig_h=None,
        image_indices=None,
    ):
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
        image_embeds = self.add_tile_position_embeds(
            image_embeds=image_embeds,
            tile_x=tile_x,
            tile_y=tile_y,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
        )

        if num_objects is None:
            return self.forward_grounding(image_embeds=image_embeds)

        dummy_targets = torch.zeros(
            pixel_values.shape[0],
            num_objects,
            4,
            dtype=torch.float32,
            device=pixel_values.device,
        )
        output = self.forward_grounding(image_embeds=image_embeds, bbox_targets=dummy_targets)
        output["grounding_loss"] = None
        output["bbox_loss"] = None
        output["mask_loss"] = None
        return output

    def predict_full_image_grounding(self, image_embeds, num_objects=None):
        if self._last_tile_infos is None:
            raise ValueError("Full-image grounding prediction requires internal tile metadata.")
        image_embeds = image_embeds.float()
        batch_size = image_embeds.shape[0]
        num_objects = min(num_objects or self.max_detection_slots, self.max_detection_slots)
        cls_embed = image_embeds[:, 0]
        queries = self.object_queries.weight[:num_objects].unsqueeze(0).expand(batch_size, -1, -1)
        object_embeds = queries + cls_embed.unsqueeze(1)

        bbox_unit_preds = torch.sigmoid(self.bbox_head(object_embeds))
        bbox_preds = bbox_unit_preds * float(self.width)
        mask_logits = image_embeds.new_zeros(batch_size, num_objects, self.height, self.width)
        tile_meta = []

        for batch_idx in range(batch_size):
            infos = self._last_tile_infos[batch_idx]
            tile_meta.append([])
            if not infos:
                continue
            for obj_idx in range(num_objects):
                info = infos[min(obj_idx, len(infos) - 1)]
                patch_embeds = self.mask_patch_head(info["embeds"][1:].float())
                patch_count = patch_embeds.shape[0]
                grid_size = int(patch_count ** 0.5)
                if grid_size * grid_size != patch_count:
                    continue
                patch_grid = patch_embeds.reshape(grid_size, grid_size, -1)
                slot_mask_logits = torch.einsum("d,hwd->hw", object_embeds[batch_idx, obj_idx], patch_grid)
                mask_logits[batch_idx, obj_idx] = F.interpolate(
                    slot_mask_logits.unsqueeze(0).unsqueeze(0),
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]
                tile_meta[batch_idx].append({
                    key: value for key, value in info.items()
                    if key != "embeds"
                })

        return {
            "bbox_preds": bbox_preds,
            "mask_logits": mask_logits,
            "tile_meta": tile_meta,
            "grounding_loss": None,
            "bbox_loss": None,
            "mask_loss": None,
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
        tile_x=None,
        tile_y=None,
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
            tile_x=tile_x,
            tile_y=tile_y,
            orig_w=orig_w,
            orig_h=orig_h,
            image_indices=image_indices,
            regions=regions,
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
        output = {
            "loss": loss,
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
            if loss is None:
                output["loss"] = detection_output["grounding_loss"]
            else:
                output["loss"] = loss + detection_output["grounding_loss"]
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
            if loss is None:
                output["loss"] = detection_output["grounding_loss"]
            else:
                output["loss"] = loss + detection_output["grounding_loss"]

        return output

    def _intersect_bbox(self, bbox, x0, y0):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        ix1 = max(x1, x0)
        iy1 = max(y1, y0)
        ix2 = min(x2, x0 + self.width)
        iy2 = min(y2, y0 + self.height)
        if ix1 >= ix2 or iy1 >= iy2:
            return None
        return [ix1 - x0, iy1 - y0, ix2 - x0, iy2 - y0]

    def _best_tile_for_region(self, infos, region):
        best_info = None
        best_bbox = None
        best_area = 0
        image_index = int(region.get("image_index", 0))
        for info in infos:
            if int(info["image_index"]) != image_index:
                continue
            local_bbox = self._intersect_bbox(region["bbox"], int(info["x0"]), int(info["y0"]))
            if local_bbox is None:
                continue
            area = max(local_bbox[2] - local_bbox[0], 0) * max(local_bbox[3] - local_bbox[1], 0)
            if area > best_area:
                best_area = area
                best_info = info
                best_bbox = local_bbox
        return best_info, best_bbox

    def _crop_mask_for_tile(self, masks, region, info, device):
        mask_index = int(region.get("mask_index", region.get("region_index", 0)))
        if mask_index >= len(masks):
            return None
        mask = masks[mask_index].to(device=device, dtype=torch.float32)
        if mask.ndim == 3:
            mask = mask[0]
        height, width = mask.shape
        pad_h = max(self.height - height, 0)
        pad_w = max(self.width - width, 0)
        if pad_h or pad_w:
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
        x0 = int(info["x0"])
        y0 = int(info["y0"])
        mask_tile = mask[y0:y0 + self.height, x0:x0 + self.width]
        if mask_tile.shape != (self.height, self.width):
            mask_tile = F.pad(mask_tile, (0, self.width - mask_tile.shape[1], 0, self.height - mask_tile.shape[0]))
        return (mask_tile > 0).float()

    def mask_criterion(self, logits, targets):
        targets = targets.float()
        positives = targets.sum()
        negatives = targets.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).clamp(min=1.0, max=100.0)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

        probs = torch.sigmoid(logits)
        smooth = 1.0
        reduce_dims = tuple(range(1, probs.ndim))
        intersection = (probs * targets).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + targets.sum(dim=reduce_dims)
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (denominator + smooth))
        dice_loss = dice_loss.mean()

        return bce_loss + dice_loss, bce_loss, dice_loss

    def forward_full_image_grounding(self, image_embeds, masks, regions, object_embeds=None):
        if self._last_tile_infos is None:
            raise ValueError("Full-image grounding requires internal tile metadata.")

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
            infos = self._last_tile_infos[batch_idx]
            for obj_idx, region in enumerate(batch_regions[:num_objects]):
                tile_info, local_bbox = self._best_tile_for_region(infos, region)
                if tile_info is None or local_bbox is None:
                    continue
                mask_tile = self._crop_mask_for_tile(masks[batch_idx], region, tile_info, device)
                if mask_tile is None:
                    continue

                patch_embeds = tile_info["embeds"][1:].float()
                patch_embeds = self.mask_patch_head(patch_embeds)
                patch_count = patch_embeds.shape[0]
                grid_size = int(patch_count ** 0.5)
                if grid_size * grid_size != patch_count:
                    continue
                patch_grid = patch_embeds.reshape(grid_size, grid_size, -1)
                slot_mask_logits = torch.einsum("d,hwd->hw", object_embeds[batch_idx, obj_idx], patch_grid)
                slot_mask_logits = F.interpolate(
                    slot_mask_logits.unsqueeze(0).unsqueeze(0),
                    size=(self.height, self.width),
                    mode="bilinear",
                    align_corners=False,
                )[0, 0]

                bbox_targets[batch_idx, obj_idx] = torch.tensor(local_bbox, dtype=torch.float32, device=device)
                mask_targets[batch_idx, obj_idx] = mask_tile
                mask_logits[batch_idx, obj_idx] = slot_mask_logits
                object_mask[batch_idx, obj_idx] = True

        grounding_loss = image_embeds.new_tensor(0.0)
        bbox_loss = None
        mask_loss = None
        mask_bce_loss = None
        mask_dice_loss = None
        if object_mask.any():
            bbox_unit_targets = bbox_targets / float(self.width)
            bbox_loss = F.l1_loss(bbox_unit_preds[object_mask], bbox_unit_targets[object_mask])
            mask_loss, mask_bce_loss, mask_dice_loss = self.mask_criterion(
                mask_logits[object_mask],
                mask_targets[object_mask],
            )
            grounding_loss = bbox_loss + mask_loss

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
            object_embeds[batch_idx, :seg_positions.numel()] = decoder_hidden_states[
                batch_idx,
                seg_positions,
            ].float()

        return object_embeds

    def forward_grounding(self, image_embeds, bbox_targets=None, mask_targets=None, object_mask=None, object_embeds=None):
        image_embeds = image_embeds.float()
        batch_size = image_embeds.shape[0]
        cls_embed = image_embeds[:, 0]
        patch_embeds = image_embeds[:, 1:]

        num_objects = self._target_slot_count(bbox_targets, mask_targets)

        if object_embeds is None:
            queries = self.object_queries.weight[:num_objects].unsqueeze(0).expand(batch_size, -1, -1)
            object_embeds = queries + cls_embed.unsqueeze(1)
        else:
            object_embeds = object_embeds[:, :num_objects].float()

        bbox_unit_preds = torch.sigmoid(self.bbox_head(object_embeds))
        bbox_preds = bbox_unit_preds * float(self.width)

        patch_embeds = self.mask_patch_head(patch_embeds)
        patch_count = patch_embeds.shape[1]
        grid_size = int(patch_count ** 0.5)
        if grid_size * grid_size != patch_count:
            raise ValueError(f"Expected square patch grid, got {patch_count} patches")

        patch_grid = patch_embeds.reshape(batch_size, grid_size, grid_size, -1)
        mask_logits = torch.einsum("bqd,bhwd->bqhw", object_embeds, patch_grid)
        mask_logits = F.interpolate(
            mask_logits,
            size=(self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )

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
            bbox_loss = F.l1_loss(bbox_unit_preds[object_mask], bbox_unit_targets[object_mask])
            grounding_loss = grounding_loss + bbox_loss

        if mask_targets is not None and object_mask.any():
            mask_targets = mask_targets[:, :num_objects].to(image_embeds.device).float()
            mask_loss, mask_bce_loss, mask_dice_loss = self.mask_criterion(
                mask_logits[object_mask],
                mask_targets[object_mask],
            )
            grounding_loss = grounding_loss + mask_loss

        return {
            "bbox_preds": bbox_preds,
            "mask_logits": mask_logits,
            "grounding_loss": grounding_loss,
            "bbox_loss": bbox_loss,
            "mask_loss": mask_loss,
            "mask_bce_loss": mask_bce_loss,
            "mask_dice_loss": mask_dice_loss,
        }

if __name__ == '__main__':
    encoder = "NorskRegnesentralSTI/NCS-v1-2d-base"  # 2d variants for seismic feature extraction trained on real field data
    decoder = "HuggingFaceTB/SmolLM-360M-Instruct"  # some random llm that fit on my laptop

    tokenizer = AutoProcessor.from_pretrained(decoder)
    model = VLM.from_encoder_decoder_pretrained(encoder, decoder, tokenizer)
