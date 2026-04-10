import json
import math
import os
import shutil
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import HTTPException


@dataclass
class InferenceConfig:
    out_dir: str = "./outputs_candidate_generation"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    hu_min: int = -1000
    hu_max: int = 400
    min_lung_voxels: int = 5000
    crop_margin_xy: int = 12
    crop_margin_z: int = 4

    stack_depth: int = 7
    gen_patch_size: int = 128
    cls_patch_size: int = 64
    gen_base_channels: int = 16
    cls_base_channels: int = 16

    proposal_threshold: float = 0.30
    proposal_topk_per_case: int = 80
    nms_distance_mm: float = 5.0
    final_cls_threshold: float = 0.50

    match_distance_mm_floor: float = 6.0


CFG: Optional[InferenceConfig] = None
MODELS = None
INIT_ERROR: Optional[str] = None


def load_saved_config(base_out_dir: str) -> InferenceConfig:
    cfg = InferenceConfig(out_dir=base_out_dir)
    cfg_path = Path(base_out_dir) / "config.json"
    if not cfg_path.exists():
        return cfg

    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    cfg_fields = set(InferenceConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in raw.items() if k in cfg_fields}
    filtered["out_dir"] = base_out_dir
    filtered["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    merged = asdict(cfg)
    merged.update(filtered)
    return InferenceConfig(**merged)


def get_loaded_config() -> InferenceConfig:
    global CFG
    if CFG is None:
        weights_dir = os.getenv("MODEL_OUT_DIR", "./outputs_candidate_generation")
        CFG = load_saved_config(weights_dir)
    return CFG


def models_ready() -> bool:
    return MODELS is not None and INIT_ERROR is None


def require_models_ready() -> None:
    if not models_ready():
        detail = INIT_ERROR or "Models are not initialized"
        raise HTTPException(status_code=503, detail=detail)


def get_service_status() -> Dict:
    cfg = get_loaded_config()
    weights_dir = Path(cfg.out_dir)
    return {
        "status": "ok",
        "device": cfg.device,
        "weights_dir": cfg.out_dir,
        "generator_weights_exists": (weights_dir / "candidate_generator.pt").exists(),
        "classifier_weights_exists": (weights_dir / "candidate_classifier.pt").exists(),
        "config_exists": (weights_dir / "config.json").exists(),
        "models_initialized": MODELS is not None,
        "model_ready": models_ready(),
        "init_error": INIT_ERROR,
    }


def sitk_load(path: str) -> sitk.Image:
    return sitk.ReadImage(path)


def image_to_np_zyx(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img)


def world_xyz_to_voxel_xyz(
    img: sitk.Image, world_xyz: Sequence[float]
) -> Tuple[float, float, float]:
    idx = img.TransformPhysicalPointToContinuousIndex(
        tuple(float(v) for v in world_xyz)
    )
    return float(idx[0]), float(idx[1]), float(idx[2])


def voxel_xyz_to_world_xyz(
    img: sitk.Image, voxel_xyz: Sequence[float]
) -> Tuple[float, float, float]:
    pt = img.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in voxel_xyz))
    return float(pt[0]), float(pt[1]), float(pt[2])


def clip_and_normalize_hu(vol_zyx: np.ndarray, hu_min: int, hu_max: int) -> np.ndarray:
    vol = np.clip(vol_zyx, hu_min, hu_max).astype(np.float32)
    vol = (vol - hu_min) / float(hu_max - hu_min)
    return vol


def zyx_bbox_from_mask(
    mask_zyx: np.ndarray,
) -> Optional[Tuple[int, int, int, int, int, int]]:
    pts = np.argwhere(mask_zyx > 0)
    if len(pts) == 0:
        return None
    z0, y0, x0 = pts.min(axis=0).tolist()
    z1, y1, x1 = pts.max(axis=0).tolist()
    return z0, z1, y0, y1, x0, x1


def expand_bbox(
    bbox: Tuple[int, int, int, int, int, int],
    shape_zyx: Sequence[int],
    margin_z: int = 0,
    margin_yx: int = 0,
) -> Tuple[int, int, int, int, int, int]:
    z0, z1, y0, y1, x0, x1 = bbox
    Z, Y, X = shape_zyx
    z0 = max(0, z0 - margin_z)
    z1 = min(Z - 1, z1 + margin_z)
    y0 = max(0, y0 - margin_yx)
    y1 = min(Y - 1, y1 + margin_yx)
    x0 = max(0, x0 - margin_yx)
    x1 = min(X - 1, x1 + margin_yx)
    return z0, z1, y0, y1, x0, x1


def crop_zyx(arr: np.ndarray, bbox: Sequence[int]) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = bbox
    return arr[z0 : z1 + 1, y0 : y1 + 1, x0 : x1 + 1]


def get_stack_indices(center_z: int, depth_z: int, stack_depth: int) -> List[int]:
    half = stack_depth // 2
    idxs = []
    for dz in range(-half, half + 1):
        z = center_z + dz
        z = min(max(z, 0), depth_z - 1)
        idxs.append(z)
    return idxs


def safe_crop_2d(
    arr2d: np.ndarray, cy: int, cx: int, patch: int, fill_value: float = 0.0
) -> np.ndarray:
    h, w = arr2d.shape
    half = patch // 2
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    out = np.full((patch, patch), fill_value, dtype=arr2d.dtype)

    sy0 = max(0, y0)
    sy1 = min(h, y1)
    sx0 = max(0, x0)
    sx1 = min(w, x1)

    dy0 = sy0 - y0
    dy1 = dy0 + (sy1 - sy0)
    dx0 = sx0 - x0
    dx1 = dx0 + (sx1 - sx0)

    out[dy0:dy1, dx0:dx1] = arr2d[sy0:sy1, sx0:sx1]
    return out


def build_spherical_mask_and_heatmap(
    shape_zyx: Sequence[int],
    spacing_xyz: Sequence[float],
    annotations_local_xyz: List[List[float]],
    diameters_mm: List[float],
    sigma_mm: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray]:
    Z, Y, X = shape_zyx
    mask = np.zeros((Z, Y, X), dtype=np.uint8)
    heatmap = np.zeros((Z, Y, X), dtype=np.float32)

    zz, yy, xx = np.indices((Z, Y, X), dtype=np.float32)
    sx, sy, sz = spacing_xyz

    for (cx, cy, cz), d_mm in zip(annotations_local_xyz, diameters_mm):
        radius_mm = max(float(d_mm) / 2.0, 1.0)
        dx_mm = (xx - cx) * sx
        dy_mm = (yy - cy) * sy
        dz_mm = (zz - cz) * sz
        dist2_mm = dx_mm**2 + dy_mm**2 + dz_mm**2
        mask |= (dist2_mm <= radius_mm**2).astype(np.uint8)

        sigma2 = max(float(sigma_mm), 1.0) ** 2
        blob = np.exp(-0.5 * dist2_mm / sigma2)
        heatmap = np.maximum(heatmap, blob.astype(np.float32))

    return mask, heatmap


def rough_body_or_lung_mask(ct_zyx: np.ndarray) -> np.ndarray:
    mask = (ct_zyx > -800).astype(np.uint8)
    per_slice = []

    for z in range(mask.shape[0]):
        comp = sitk.GetImageFromArray(mask[z].astype(np.uint8))
        comp = sitk.BinaryMorphologicalClosing(comp, [4, 4])
        comp = sitk.BinaryFillhole(comp)
        arr = sitk.GetArrayFromImage(comp)[0]
        per_slice.append(arr)

    out = np.stack(per_slice, axis=0).astype(np.uint8)
    if out.sum() == 0:
        out[:] = 1
    return out


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet2p5D(nn.Module):
    def __init__(self, in_ch: int, base: int = 16):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


class TinyEncoderClassifier(nn.Module):
    def __init__(self, in_ch: int, base: int = 16):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch, base),
            nn.MaxPool2d(2),
            ConvBlock(base, base * 2),
            nn.MaxPool2d(2),
            ConvBlock(base * 2, base * 4),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(base * 4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        return self.head(x)


class ModelBundle:
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.generator = SmallUNet2p5D(cfg.stack_depth, cfg.gen_base_channels).to(
            self.device
        )
        self.classifier = TinyEncoderClassifier(
            cfg.stack_depth, cfg.cls_base_channels
        ).to(self.device)
        self.generator.eval()
        self.classifier.eval()
        self._load_weights()

    def _load_weights(self) -> None:
        out_dir = Path(self.cfg.out_dir)
        generator_path = out_dir / "candidate_generator.pt"
        classifier_path = out_dir / "candidate_classifier.pt"

        if not generator_path.exists():
            raise FileNotFoundError(f"Generator weights not found: {generator_path}")
        if not classifier_path.exists():
            raise FileNotFoundError(f"Classifier weights not found: {classifier_path}")

        gen_state = torch.load(str(generator_path), map_location=self.device)
        cls_state = torch.load(str(classifier_path), map_location=self.device)

        self.generator.load_state_dict(gen_state)
        self.classifier.load_state_dict(cls_state)


def summarize_candidate_coordinates(candidates: List[Dict]) -> List[Dict]:
    rows = []
    for i, c in enumerate(candidates, start=1):
        rows.append(
            {
                "candidate_index": i,
                "is_positive_candidate": bool(c["is_positive_candidate"]),
                "proposal_score": float(c["proposal_score"]),
                "cls_score": float(c["cls_score"]),
                "voxel_xyz_local": [int(v) for v in c["voxel_xyz_local"]],
                "voxel_xyz_global": [int(v) for v in c["voxel_xyz_global"]],
                "world_xyz": [float(v) for v in c["world_xyz"]],
                "x": int(c["voxel_xyz_global"][0]),
                "y": int(c["voxel_xyz_global"][1]),
                "slice_z": int(c["voxel_xyz_global"][2]),
            }
        )
    return rows


@torch.no_grad()
def predict_case_heatmap(
    generator: nn.Module, case_meta: Dict, cfg: InferenceConfig
) -> np.ndarray:
    generator.eval()
    vol = case_meta["ct_zyx"]
    Z, Y, X = vol.shape
    device = next(generator.parameters()).device
    pred = np.zeros((Z, Y, X), dtype=np.float32)

    patch = cfg.gen_patch_size
    half = patch // 2

    y_centers = list(range(half, max(half + 1, Y - half + 1), half))
    x_centers = list(range(half, max(half + 1, X - half + 1), half))

    if y_centers[-1] != Y - half:
        y_centers.append(max(half, Y - half))
    if x_centers[-1] != X - half:
        x_centers.append(max(half, X - half))

    count = np.zeros_like(pred)

    for z in range(Z):
        z_idxs = get_stack_indices(z, Z, cfg.stack_depth)
        for cy in y_centers:
            for cx in x_centers:
                stack = [
                    safe_crop_2d(vol[zz], cy, cx, patch, fill_value=0.0)
                    for zz in z_idxs
                ]
                x_tensor = (
                    torch.from_numpy(np.stack(stack, axis=0)[None]).float().to(device)
                )
                logits = generator(x_tensor)
                probs = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()

                y0 = cy - half
                x0 = cx - half
                y1 = min(Y, y0 + patch)
                x1 = min(X, x0 + patch)
                py1 = y1 - y0
                px1 = x1 - x0

                pred[z, y0:y1, x0:x1] += probs[:py1, :px1]
                count[z, y0:y1, x0:x1] += 1.0

    return pred / np.maximum(count, 1.0)


def extract_candidate_peaks(
    heatmap_zyx: np.ndarray, case_meta: Dict, cfg: InferenceConfig
) -> List[Dict]:
    x = torch.from_numpy(heatmap_zyx[None, None]).float()
    pooled = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    is_peak = (x == pooled) & (x >= cfg.proposal_threshold)
    coords = torch.nonzero(is_peak[0, 0], as_tuple=False).cpu().numpy()

    scores = []
    for z, y, x_ in coords:
        scores.append((float(heatmap_zyx[z, y, x_]), int(z), int(y), int(x_)))

    scores.sort(reverse=True)
    scores = scores[: cfg.proposal_topk_per_case]

    spacing_xyz = case_meta["spacing_xyz"]
    kept: List[Dict] = []
    ct_img = case_meta["ct_img"]

    for score, z, y, x_ in scores:
        keep = True

        for prev in kept:
            dx = (x_ - prev["voxel_xyz_local"][0]) * spacing_xyz[0]
            dy = (y - prev["voxel_xyz_local"][1]) * spacing_xyz[1]
            dz = (z - prev["voxel_xyz_local"][2]) * spacing_xyz[2]
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < cfg.nms_distance_mm:
                keep = False
                break

        if keep:
            bbox = case_meta["bbox_zyx"]
            global_xyz = [int(x_ + bbox[4]), int(y + bbox[2]), int(z + bbox[0])]
            world_xyz = list(voxel_xyz_to_world_xyz(ct_img, global_xyz))

            kept.append(
                {
                    "seriesuid": case_meta["seriesuid"],
                    "proposal_score": float(score),
                    "voxel_xyz_local": [int(x_), int(y), int(z)],
                    "voxel_xyz_global": global_xyz,
                    "world_xyz": world_xyz,
                }
            )

    return kept


@torch.no_grad()
def classify_candidates_for_case(
    classifier: nn.Module,
    case_meta: Dict,
    candidates: List[Dict],
    cfg: InferenceConfig,
) -> List[Dict]:
    classifier.eval()
    device = next(classifier.parameters()).device
    vol = case_meta["ct_zyx"]
    results = []

    for c in candidates:
        x, y, z = c["voxel_xyz_local"]
        z_idxs = get_stack_indices(int(z), vol.shape[0], cfg.stack_depth)
        stack = [
            safe_crop_2d(vol[zz], int(y), int(x), cfg.cls_patch_size, fill_value=0.0)
            for zz in z_idxs
        ]
        inp = torch.from_numpy(np.stack(stack, axis=0)[None]).float().to(device)
        prob = torch.sigmoid(classifier(inp)).item()

        row = dict(c)
        row["cls_score"] = float(prob)
        row["is_positive_candidate"] = bool(prob >= cfg.final_cls_threshold)
        results.append(row)

    results.sort(key=lambda d: d["cls_score"], reverse=True)
    return results


def dice_binary(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    inter = float((pred * target).sum())
    denom = float(pred.sum() + target.sum())
    return float((2.0 * inter + eps) / (denom + eps))


def euclidean_mm(
    local_xyz_a: Sequence[float],
    local_xyz_b: Sequence[float],
    spacing_xyz: Sequence[float],
) -> float:
    dx = (float(local_xyz_a[0]) - float(local_xyz_b[0])) * float(spacing_xyz[0])
    dy = (float(local_xyz_a[1]) - float(local_xyz_b[1])) * float(spacing_xyz[1])
    dz = (float(local_xyz_a[2]) - float(local_xyz_b[2])) * float(spacing_xyz[2])
    return float(math.sqrt(dx * dx + dy * dy + dz * dz))


def match_candidates_to_annotations(
    final_candidates: List[Dict],
    annotations_local_xyz: List[List[float]],
    diameters_mm: List[float],
    spacing_xyz: Sequence[float],
    cfg: InferenceConfig,
) -> Dict:
    preds = [c for c in final_candidates if c["is_positive_candidate"]]
    matched_pred_idx = set()
    matched_ann_idx = set()

    for ann_idx, ann_xyz in enumerate(annotations_local_xyz):
        best_pred_idx = None
        best_dist = float("inf")

        match_radius_mm = (
            max(cfg.match_distance_mm_floor, float(diameters_mm[ann_idx]) / 2.0)
            if ann_idx < len(diameters_mm)
            else cfg.match_distance_mm_floor
        )

        for pred_idx, pred in enumerate(preds):
            if pred_idx in matched_pred_idx:
                continue

            dist_mm = euclidean_mm(
                pred["voxel_xyz_local"],
                ann_xyz,
                spacing_xyz,
            )

            if dist_mm <= match_radius_mm and dist_mm < best_dist:
                best_dist = dist_mm
                best_pred_idx = pred_idx

        if best_pred_idx is not None:
            matched_pred_idx.add(best_pred_idx)
            matched_ann_idx.add(ann_idx)

    tp = len(matched_pred_idx)
    fp = max(0, len(preds) - tp)
    fn = max(0, len(annotations_local_xyz) - len(matched_ann_idx))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    proposal_recall_hits = 0
    for ann_idx, ann_xyz in enumerate(annotations_local_xyz):
        match_radius_mm = (
            max(cfg.match_distance_mm_floor, float(diameters_mm[ann_idx]) / 2.0)
            if ann_idx < len(diameters_mm)
            else cfg.match_distance_mm_floor
        )
        hit = any(
            euclidean_mm(c["voxel_xyz_local"], ann_xyz, spacing_xyz) <= match_radius_mm
            for c in final_candidates
        )
        proposal_recall_hits += int(hit)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "proposal_stage_recall": float(
            proposal_recall_hits / max(len(annotations_local_xyz), 1)
        ),
    }


def read_dicom_series(dicom_dir: str) -> sitk.Image:
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(dicom_dir)
    if not series_ids:
        raise ValueError(f"No DICOM series found in: {dicom_dir}")

    series_id = series_ids[0]
    file_names = reader.GetGDCMSeriesFileNames(dicom_dir, series_id)
    if not file_names:
        raise ValueError(f"No DICOM files found for series {series_id} in: {dicom_dir}")

    reader.SetFileNames(file_names)
    return reader.Execute()


def preprocess_external_case(
    ct_img: sitk.Image,
    cfg: InferenceConfig,
    lung_mask_img: Optional[sitk.Image] = None,
    annotations_world_xyz: Optional[List[List[float]]] = None,
    annotation_diameters_mm: Optional[List[float]] = None,
    seriesuid: Optional[str] = None,
) -> Dict:
    ct_zyx = image_to_np_zyx(ct_img).astype(np.float32)

    if lung_mask_img is not None:
        lung_zyx = (image_to_np_zyx(lung_mask_img) > 0).astype(np.uint8)
    else:
        lung_zyx = rough_body_or_lung_mask(ct_zyx)

    bbox = zyx_bbox_from_mask(lung_zyx)
    if bbox is None:
        bbox = (
            0,
            ct_zyx.shape[0] - 1,
            0,
            ct_zyx.shape[1] - 1,
            0,
            ct_zyx.shape[2] - 1,
        )

    bbox = expand_bbox(
        bbox,
        ct_zyx.shape,
        margin_z=cfg.crop_margin_z,
        margin_yx=cfg.crop_margin_xy,
    )

    ct_crop = crop_zyx(ct_zyx, bbox)
    lung_crop = crop_zyx(lung_zyx, bbox)

    if int(lung_crop.sum()) < cfg.min_lung_voxels:
        lung_crop = np.ones_like(ct_crop, dtype=np.uint8)

    ct_norm = clip_and_normalize_hu(ct_crop, cfg.hu_min, cfg.hu_max)

    z0, _, y0, _, x0, _ = bbox
    ann_world = annotations_world_xyz or []
    ann_diams = annotation_diameters_mm or []
    anns_local: List[List[float]] = []

    for world_xyz in ann_world:
        vx, vy, vz = world_xyz_to_voxel_xyz(ct_img, world_xyz)
        local_xyz = [vx - x0, vy - y0, vz - z0]

        if (
            0 <= local_xyz[0] < ct_crop.shape[2]
            and 0 <= local_xyz[1] < ct_crop.shape[1]
            and 0 <= local_xyz[2] < ct_crop.shape[0]
        ):
            anns_local.append(local_xyz)

    if len(ann_diams) != len(anns_local):
        ann_diams = ann_diams[: len(anns_local)] + [6.0] * max(
            0, len(anns_local) - len(ann_diams)
        )

    _, center_heatmap = build_spherical_mask_and_heatmap(
        shape_zyx=ct_crop.shape,
        spacing_xyz=tuple(float(v) for v in ct_img.GetSpacing()),
        annotations_local_xyz=anns_local,
        diameters_mm=ann_diams,
        sigma_mm=4.0,
    )

    return {
        "seriesuid": seriesuid or "external_case",
        "ct_img": ct_img,
        "bbox_zyx": [int(v) for v in bbox],
        "shape_zyx": [int(v) for v in ct_crop.shape],
        "spacing_xyz": [float(v) for v in ct_img.GetSpacing()],
        "origin_xyz": [float(v) for v in ct_img.GetOrigin()],
        "direction": [float(v) for v in ct_img.GetDirection()],
        "ct_zyx": ct_norm.astype(np.float32),
        "lung_zyx": lung_crop.astype(np.uint8),
        "center_heatmap_zyx": center_heatmap.astype(np.float32),
        "annotations_local_xyz": [[float(v) for v in p] for p in anns_local],
        "annotation_diameters_mm": [float(v) for v in ann_diams],
        "num_annotations": len(anns_local),
    }


def run_case(case_meta: Dict) -> Dict:
    require_models_ready()
    pred_heatmap = predict_case_heatmap(MODELS.generator, case_meta, CFG)
    proposals = extract_candidate_peaks(pred_heatmap, case_meta, CFG)
    final_candidates = classify_candidates_for_case(
        MODELS.classifier, case_meta, proposals, CFG
    )

    metrics = None
    target_heatmap = case_meta.get("center_heatmap_zyx")
    has_target = (
        target_heatmap is not None
        and float(target_heatmap.max()) > 0
        and int(case_meta.get("num_annotations", 0)) > 0
    )

    if has_target:
        metrics = {
            "heatmap_dice_at_0_5": dice_binary(
                (pred_heatmap >= 0.5).astype(np.uint8),
                (target_heatmap >= 0.5).astype(np.uint8),
            ),
            "heatmap_target_max": float(target_heatmap.max()),
            "num_annotations": int(case_meta.get("num_annotations", 0)),
        }
        metrics.update(
            match_candidates_to_annotations(
                final_candidates=final_candidates,
                annotations_local_xyz=case_meta.get("annotations_local_xyz", []),
                diameters_mm=case_meta.get("annotation_diameters_mm", []),
                spacing_xyz=case_meta["spacing_xyz"],
                cfg=CFG,
            )
        )

    positives = [c for c in final_candidates if c["is_positive_candidate"]]
    negatives = [c for c in final_candidates if not c["is_positive_candidate"]]

    return {
        "seriesuid": case_meta["seriesuid"],
        "shape_zyx": case_meta["shape_zyx"],
        "spacing_xyz": case_meta["spacing_xyz"],
        "num_proposals": len(proposals),
        "num_final_candidates": len(final_candidates),
        "num_positive_candidates": len(positives),
        "num_negative_candidates": len(negatives),
        "positive_candidates": summarize_candidate_coordinates(positives),
        "negative_candidates": summarize_candidate_coordinates(negatives),
        "all_candidates": summarize_candidate_coordinates(final_candidates),
        "metrics": metrics,
    }


def initialize_models() -> None:
    global CFG, MODELS, INIT_ERROR
    weights_dir = os.getenv("MODEL_OUT_DIR", "./outputs_candidate_generation")
    CFG = load_saved_config(weights_dir)

    try:
        MODELS = ModelBundle(CFG)
        INIT_ERROR = None
    except Exception as e:
        MODELS = None
        INIT_ERROR = str(e)


def predict_from_volume_path(
    ct_path: str,
    lung_mask_path: Optional[str],
    annotations_world_xyz: Optional[List[List[float]]],
    annotation_diameters_mm: Optional[List[float]],
    seriesuid: Optional[str],
) -> Dict:
    require_models_ready()
    cfg = get_loaded_config()

    ct_img = sitk_load(ct_path)
    lung_mask_img = sitk_load(lung_mask_path) if lung_mask_path else None

    case_meta = preprocess_external_case(
        ct_img=ct_img,
        cfg=cfg,
        lung_mask_img=lung_mask_img,
        annotations_world_xyz=annotations_world_xyz,
        annotation_diameters_mm=annotation_diameters_mm,
        seriesuid=seriesuid or Path(ct_path).stem,
    )
    return run_case(case_meta)


def predict_from_dicom_dir(
    dicom_dir: str,
    lung_mask_path: Optional[str],
    annotations_world_xyz: Optional[List[List[float]]],
    annotation_diameters_mm: Optional[List[float]],
    seriesuid: Optional[str],
) -> Dict:
    require_models_ready()
    cfg = get_loaded_config()

    try:
        ct_img = read_dicom_series(dicom_dir)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read DICOM series: {e}")

    lung_mask_img = sitk_load(lung_mask_path) if lung_mask_path else None

    case_meta = preprocess_external_case(
        ct_img=ct_img,
        cfg=cfg,
        lung_mask_img=lung_mask_img,
        annotations_world_xyz=annotations_world_xyz,
        annotation_diameters_mm=annotation_diameters_mm,
        seriesuid=seriesuid or Path(dicom_dir).name,
    )
    return run_case(case_meta)


def _find_single_volume_file(root_dir: Path) -> Path:
    allowed = [".nii.gz", ".nii", ".mhd", ".mha"]

    files: List[Path] = []
    for p in root_dir.rglob("*"):
        if not p.is_file():
            continue
        name_lower = p.name.lower()
        if any(name_lower.endswith(ext) for ext in allowed):
            files.append(p)

    if not files:
        raise HTTPException(
            status_code=400, detail="No supported medical volume file found in upload"
        )

    if len(files) > 1:
        # Prefer .nii.gz first, then .nii, then .mhd, then .mha
        def sort_key(p: Path) -> Tuple[int, str]:
            name = p.name.lower()
            if name.endswith(".nii.gz"):
                rank = 0
            elif name.endswith(".nii"):
                rank = 1
            elif name.endswith(".mhd"):
                rank = 2
            else:
                rank = 3
            return (rank, str(p))

        files.sort(key=sort_key)

    return files[0]


def _find_dicom_root(extract_dir: Path) -> Path:
    try:
        read_dicom_series(str(extract_dir))
        return extract_dir
    except Exception:
        pass

    for subdir in sorted([p for p in extract_dir.rglob("*") if p.is_dir()]):
        try:
            read_dicom_series(str(subdir))
            return subdir
        except Exception:
            continue

    raise HTTPException(
        status_code=400, detail="No readable DICOM study found in uploaded zip"
    )


def predict_from_uploaded_volume_bytes(
    file_bytes: bytes,
    filename: str,
    annotations_world_xyz: Optional[List[List[float]]],
    annotation_diameters_mm: Optional[List[float]],
    seriesuid: Optional[str],
) -> Dict:
    require_models_ready()

    suffix = "".join(Path(filename).suffixes).lower()
    allowed = {".nii", ".nii.gz", ".mha", ".mhd"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported upload type: {suffix}. Use one of {sorted(allowed)}",
        )

    temp_dir = tempfile.mkdtemp(prefix="ct_upload_")
    try:
        temp_path = Path(temp_dir) / filename
        with open(temp_path, "wb") as f:
            f.write(file_bytes)

        return predict_from_volume_path(
            ct_path=str(temp_path),
            lung_mask_path=None,
            annotations_world_xyz=annotations_world_xyz,
            annotation_diameters_mm=annotation_diameters_mm,
            seriesuid=seriesuid or Path(filename).stem,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def predict_from_uploaded_dicom_zip_bytes(
    zip_bytes: bytes,
    filename: str,
    annotations_world_xyz: Optional[List[List[float]]],
    annotation_diameters_mm: Optional[List[float]],
    seriesuid: Optional[str],
) -> Dict:
    if not filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="DICOM upload must be a .zip file")

    temp_dir = tempfile.mkdtemp(prefix="dicom_upload_")
    try:
        zip_path = Path(temp_dir) / filename
        with open(zip_path, "wb") as f:
            f.write(zip_bytes)

        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir)
        except zipfile.BadZipFile:
            raise HTTPException(
                status_code=400, detail="Uploaded file is not a valid zip archive"
            )

        require_models_ready()

        dicom_root = _find_dicom_root(extract_dir)

        return predict_from_dicom_dir(
            dicom_dir=str(dicom_root),
            lung_mask_path=None,
            annotations_world_xyz=annotations_world_xyz,
            annotation_diameters_mm=annotation_diameters_mm,
            seriesuid=seriesuid or Path(filename).stem,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
