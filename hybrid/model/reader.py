"""Instance reader — autoregressive, query-free detector over frozen NCS features.

Replaces the dense-seg + RANSAC front-end (measure-only; masks live on the <SEG>
decoder). Emits objects as a SEQUENCE (emit-until-stop → cap-free): per step a
class + a soft footprint over the feature grid + class-driven LEARNABLE attribute
heads. THE RULE (reference_dip_from_mask_geometry): angle/shape attrs read the
SPATIAL footprint (dip from its 2nd-moment stats), magnitudes read POOLED features
(throw). A dip head on a pooled scalar collapses to 73° — never do that.

Whole-object attention: the decoder attends over ALL grid tokens of a window.
Teacher-forced during training (GT objects ordered by x-centre → no Hungarian).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
NO_OBJ, FAULT, CLOSURE = 0, 1, 2          # class ids (∅ / fault / closure)


def scene_to_gt(scene):
    """Scene objs -> reader GT list {cls, ctr, mask_fW, dip?, throw?, area?}, ordered by x-centre."""
    smap = scene["smap"]; fH, fW = smap.shape[1], smap.shape[2]
    objs = []
    for o in scene["objs"]:
        c = int(o["cls"])
        if c not in (1, 2):
            continue
        x1, y1, x2, y2 = o["bbox"]
        ctr = torch.tensor([(y1 + y2) / 2, (x1 + x2) / 2], device=device, dtype=torch.float32)
        mfw = F.adaptive_avg_pool2d(o["mask"][None, None].float(), (fH, fW))[0, 0].clamp(0, 1)
        g = dict(cls=c, ctr=ctr, mask_fW=mfw, mask_full=o["mask"])
        g["dip"] = float(o["meas"][0]) if (c == 1 and float(o["mmask"][0]) > 0) else None
        g["throw"] = float(o["meas"][1]) if (c == 1 and float(o["mmask"][1]) > 0) else None
        g["area"] = float(o["meas"][2]) if (c == 2 and float(o["mmask"][2]) > 0) else None
        objs.append(((x1 + x2) / 2, g))
    objs.sort(key=lambda t: t[0])
    return [g for _, g in objs]


class InstanceReader(nn.Module):
    def __init__(self, vdim=768, d=256, layers=3, heads=8, max_steps=24, pixel_decoder=True):
        super().__init__()
        self.d, self.max_steps = d, max_steps
        self.proj = nn.Linear(vdim, d)
        # Pixel decoder: global self-attention over the STITCHED map — reassembles
        # cross-tile context that per-tile NCS encoding loses, so a tall fault split
        # across tiles becomes whole-object BEFORE the reader reads it.
        self.pixdec = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True), 2) if pixel_decoder else None
        self.pos = nn.Parameter(torch.zeros(1, 4096, d))       # grid pos-enc (flattened)
        self.bos = nn.Parameter(torch.zeros(1, 1, d))
        self.obj_cls = nn.Embedding(3, d)                       # embed a previous object's class
        self.obj_ctr = nn.Linear(2, d)                         # + its footprint centroid
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True)
        self.dec = nn.TransformerDecoder(dec, layers)
        self.stop_head = nn.Linear(d, 1)
        self.class_head = nn.Linear(d, 3)
        self.foot_q = nn.Linear(d, d)                          # pooling footprint (softmax → dip/pool)
        self.occ_q = nn.Linear(d, d)                           # occupancy footprint (sigmoid → mask/area)
        self.dip_head = nn.Sequential(nn.Linear(7, 64), nn.GELU(), nn.Linear(64, 1))    # SPATIAL stats → dip
        self.throw_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))  # POOLED → throw
        self.area_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))   # POOLED → area
        # Mask head (Mask2Former-style): instance query · UPSAMPLED pixel-decoder features →
        # per-instance mask. Shares the trunk with the reader → mask loss co-trains the pixel
        # decoder (couples facts + masks in Stage 2; no LM dependency). The query is the content prompt.
        self.mask_q = nn.Linear(d, d)
        mlrs = [nn.Conv2d(d, d, 3, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        for _ in range(3):
            mlrs += [nn.ConvTranspose2d(d, d, 4, stride=2, padding=1), nn.GroupNorm(8, d), nn.GELU()]
        self.mask_up = nn.Sequential(*mlrs, nn.Conv2d(d, d, 1))     # 8× upsample → mask features

    def _grid(self, smap):
        """smap (vdim,fH,fW) -> memory tokens (1,fHW,d), and the (row,col) coords."""
        fH, fW = smap.shape[1], smap.shape[2]
        m = self.proj(smap.flatten(1).t())                     # (fHW, d)
        m = (m + self.pos[:, : m.shape[0]].squeeze(0)).unsqueeze(0)   # (1, fHW, d)
        if self.pixdec is not None:
            m = self.pixdec(m)                                 # cross-tile global attention
        rr, cc = torch.meshgrid(torch.linspace(0, 1, fH, device=device),
                                torch.linspace(0, 1, fW, device=device), indexing="ij")
        coord = torch.stack([rr.flatten(), cc.flatten()], -1)  # (fHW, 2)
        return m, coord, (fH, fW)

    def _readout(self, h, memory, coord):
        """h (B,T,d) decoder states -> per-step heads. Footprint over the grid gives
        the spatial stats for dip; pooled feature gives throw/area."""
        logits = self.foot_q(h) @ memory.transpose(1, 2)        # (B,T,fHW) pooling scores
        w = logits.softmax(-1)                                  # pooling weights (→ dip/pool)
        occ_logits = self.occ_q(h) @ memory.transpose(1, 2)     # (B,T,fHW) occupancy (own head)
        foot = occ_logits.sigmoid()                             # per-cell occupancy (mask/area)
        pooled = w @ memory                                     # (B,T,d)
        # spatial 2nd-moment stats of the footprint -> dip (angle-preserving input)
        mu = w @ coord                                         # (B,T,2) centroid
        d0 = coord.unsqueeze(0).unsqueeze(0) - mu.unsqueeze(2)  # (B,T,fHW,2)
        cov = torch.einsum("btni,btnj,btn->btij", d0, d0, w)   # (B,T,2,2)
        stats = torch.stack([mu[..., 0], mu[..., 1], cov[..., 0, 0], cov[..., 1, 1],
                             cov[..., 0, 1], (mu[..., 0] - .5), (mu[..., 1] - .5)], -1)
        return dict(stop=self.stop_head(h).squeeze(-1), cls=self.class_head(h),
                    dip=self.dip_head(stats).squeeze(-1), throw=self.throw_head(pooled).squeeze(-1),
                    area=self.area_head(pooled).squeeze(-1), foot=foot, foot_logits=occ_logits, mu=mu)

    def _mask_features(self, memory, fH, fW):
        """Pixel-decoder trunk features (1,fHW,d) → upsampled per-pixel mask features (1,d,H',W')."""
        return self.mask_up(memory.transpose(1, 2).reshape(1, self.d, fH, fW))

    def _seq_embed(self, classes, centroids):
        """Embed a prefix of GT/emitted objects for teacher forcing / AR decode."""
        e = self.obj_cls(classes) + self.obj_ctr(centroids)    # (B,K,d)
        return torch.cat([self.bos.expand(e.shape[0], -1, -1), e], 1)  # prepend BOS

    def forward(self, smap, gt):
        """Teacher-forced loss on one scene. gt = list of dicts {cls, dip, throw?, area?,
        ctr(2), mask_fW(fH,fW), mask_full(H,W)}. Returns (loss, parts)."""
        memory, coord, (fH, fW) = self._grid(smap)
        K = len(gt)
        cls = torch.tensor([o["cls"] for o in gt], device=device).unsqueeze(0)
        ctr = torch.stack([o["ctr"] for o in gt]).unsqueeze(0) if K else torch.zeros(1, 0, 2, device=device)
        tgt = self._seq_embed(cls, ctr)                        # (1,K+1,d)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        h = self.dec(tgt, memory, tgt_mask=mask)               # (1,K+1,d); step t predicts object t
        out = self._readout(h, memory, coord)

        # supervise steps 0..K-1 = the K objects, step K = STOP
        stop_t = torch.zeros(1, K + 1, device=device); stop_t[0, K] = 1.0
        L = F.binary_cross_entropy_with_logits(out["stop"], stop_t)
        parts = {"stop": L.item()}
        if K:
            cl = F.cross_entropy(out["cls"][0, :K], cls[0])
            L = L + cl; parts["cls"] = cl.item()
            gm = torch.stack([o["mask_fW"] for o in gt]).to(device).flatten(1).clamp(0, 1)  # (K,fHW)
            pw = torch.tensor([20.0], device=device)
            occ = out["foot"][0, :K]
            fp = (F.binary_cross_entropy_with_logits(out["foot_logits"][0, :K], gm, pos_weight=pw)
                  + 1 - (2 * (occ * gm).sum() + 1) / (occ.sum() + gm.sum() + 1))    # pos-BCE + dice
            L = L + fp; parts["foot"] = fp.item()
            has_d = torch.tensor([o["cls"] == FAULT and o.get("dip") is not None for o in gt], device=device)
            if has_d.any():
                gd = torch.tensor([o.get("dip") or 0.0 for o in gt], device=device)
                dl = F.smooth_l1_loss(out["dip"][0, :K][has_d], gd[has_d] / 90.0)
                L = L + dl; parts["dip"] = dl.item()
            has_t = torch.tensor([o["cls"] == FAULT and o.get("throw") is not None for o in gt], device=device)
            if has_t.any():
                gtt = torch.tensor([o.get("throw") or 0.0 for o in gt], device=device)
                tl = F.smooth_l1_loss(out["throw"][0, :K][has_t], gtt[has_t] / 500.0)
                L = L + tl; parts["throw"] = tl.item()
            clo = torch.tensor([o["cls"] == CLOSURE and o.get("area") is not None for o in gt], device=device)
            if clo.any():
                ga = torch.tensor([o.get("area") or 0.0 for o in gt], device=device)[clo] / 100.0
                al = F.smooth_l1_loss(out["area"][0, :K][clo], ga)
                L = L + al; parts["area"] = al.item()
            mfull = [o.get("mask_full") for o in gt]
            if any(mm is not None for mm in mfull):
                mfeat = self._mask_features(memory, fH, fW)            # (1,d,H',W')
                ml = torch.einsum("kd,dhw->khw", self.mask_q(h[0, :K]), mfeat[0])  # (K,H',W')
                Ht, Wt = mfull[0].shape
                ml = F.interpolate(ml.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]
                gm2 = torch.stack([mm.to(device) for mm in mfull]).float().clamp(0, 1)
                p2 = ml.sigmoid(); pw2 = torch.tensor([40.0], device=device)
                mk = (F.binary_cross_entropy_with_logits(ml, gm2, pos_weight=pw2)
                      + 1 - (2 * (p2 * gm2).sum() + 1) / (p2.sum() + gm2.sum() + 1))
                L = L + mk; parts["mask"] = mk.item()
        return L, parts

    @torch.no_grad()
    def tf_masks(self, smap, gt):
        """Teacher-forced per-object mask logits (interp to GT mask size) — for mask eval."""
        memory, coord, (fH, fW) = self._grid(smap)
        K = len(gt)
        cls = torch.tensor([o["cls"] for o in gt], device=device).unsqueeze(0)
        ctr = torch.stack([o["ctr"] for o in gt]).unsqueeze(0)
        tgt = self._seq_embed(cls, ctr)
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        h = self.dec(tgt, memory, tgt_mask=mask)
        ml = torch.einsum("kd,dhw->khw", self.mask_q(h[0, :K]), self._mask_features(memory, fH, fW)[0])
        Ht, Wt = gt[0]["mask_full"].shape
        return F.interpolate(ml.unsqueeze(0), size=(Ht, Wt), mode="bilinear", align_corners=False)[0]

    @torch.no_grad()
    def detect(self, smap, thresh=0.5, want_masks=False):
        """Greedy autoregressive decode → list of {cls, dip, throw, area, ctr}. With
        want_masks, also returns per-instance mask logits (H',W') from the query."""
        memory, coord, (fH, fW) = self._grid(smap)
        mfeat = self._mask_features(memory, fH, fW) if want_masks else None
        objs, masks, cls_hist, ctr_hist = [], [], [], []
        for _ in range(self.max_steps):
            cls_t = torch.tensor(cls_hist, device=device).unsqueeze(0) if cls_hist else torch.zeros(1, 0, dtype=torch.long, device=device)
            ctr_t = torch.stack(ctr_hist).unsqueeze(0) if ctr_hist else torch.zeros(1, 0, 2, device=device)
            tgt = self._seq_embed(cls_t, ctr_t)
            m = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            h = self.dec(tgt, memory, tgt_mask=m)
            out = self._readout(h[:, -1:], memory, coord)
            if out["stop"][0, 0].sigmoid() > thresh:
                break
            c = int(out["cls"][0, 0].argmax())
            if c == NO_OBJ:
                break
            objs.append(dict(cls=c, dip=float(out["dip"][0, 0] * 90.0),
                             throw=float(out["throw"][0, 0] * 500.0),
                             area=float(out["area"][0, 0] * 100.0),
                             ctr=out["mu"][0, 0].detach()))
            if want_masks:
                masks.append(torch.einsum("bd,dhw->bhw", self.mask_q(h[:, -1]), mfeat[0])[0])
            cls_hist.append(c); ctr_hist.append(out["mu"][0, 0].detach())
        return (objs, masks) if want_masks else objs
