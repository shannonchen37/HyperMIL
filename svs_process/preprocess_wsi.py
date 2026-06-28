# -*- coding: utf-8 -*-
import os
import math
import glob
import time
import queue
import random
import argparse
import traceback
import threading
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager

import yaml
import numpy as np
import cv2
import openslide
from PIL import Image, ImageDraw

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DEFAULT_CONFIG_PATH = os.path.join(SCRIPT_DIR, "preprocess_config.yaml")


# =========================
# General utilities
# =========================
@contextmanager
def suppress_c_stderr():
    """Silence stderr from low-level C libraries such as libtiff or libjpeg."""
    try:
        stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 2)
        os.close(devnull)
        yield
    finally:
        try:
            os.dup2(stderr_fd, 2)
        finally:
            os.close(stderr_fd)


def check_dir(d: str):
    os.makedirs(d, exist_ok=True)


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def log(msg: str):
    tqdm.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def short_id(s: str, keep: int = 32) -> str:
    return s if len(s) <= keep else "..." + s[-keep:]


def atomic_save_npy(arr: np.ndarray, out_path: str):
    tmp = out_path + ".tmp"
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, out_path)


def safe_load_npy(path: str) -> Optional[np.ndarray]:
    """Load a NumPy file safely, renaming broken files to `.broken`."""
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        log(f"[warn] Failed to read {path}: {e}. Renaming it to .broken for recomputation.")
        try:
            os.replace(path, path + ".broken")
        except Exception:
            pass
        return None


def list_done_feature_ids(patch_ft_dir: str) -> set:
    """Infer completed slide IDs from existing `*_fts.npy` files."""
    done = set()
    for p in glob.glob(os.path.join(patch_ft_dir, "*_fts.npy")):
        base = os.path.basename(p)
        if base.endswith("_fts.npy"):
            done.add(base[:-8])
    return done


def build_wsi_paths_from_root(data_root: str,
                              exts: Tuple[str, ...] = (".svs", ".tif", ".ndpi")) -> List[str]:
    """Recursively scan `data_root` for supported slide extensions."""
    results = []
    exts_lower = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(data_root):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in exts_lower:
                results.append(os.path.join(root, fn))
    results.sort()
    return results


def slide_id_from_path(p: str) -> str:
    parent = os.path.basename(os.path.dirname(p))
    stem = os.path.splitext(os.path.basename(p))[0]
    return f"{parent}_{stem}"


# =========================
# OpenSlide helpers and visualization
# =========================
def level_dims(slide: openslide.OpenSlide, level: int) -> Tuple[int, int]:
    return slide.level_dimensions[level]


def scale_sample_to_level0(slide: openslide.OpenSlide, sample_level: int) -> Tuple[float, float]:
    Ws, Hs = level_dims(slide, sample_level)
    W0, H0 = level_dims(slide, 0)
    return W0 / float(Ws), H0 / float(Hs)


def tile_size_level0(slide: openslide.OpenSlide, sample_level: int, patch_size: int) -> Tuple[int, int]:
    sx0, sy0 = scale_sample_to_level0(slide, sample_level)
    pw0 = max(1, int(round(patch_size * sx0)))
    ph0 = max(1, int(round(patch_size * sy0)))
    return pw0, ph0


def draw_patches_on_slide_level0(slide_path: str,
                                 coors_lvl0: List[Tuple[int, int]],
                                 patch_size: int,
                                 sample_level: int,
                                 downsample: int) -> Image.Image:
    slide = openslide.open_slide(slide_path)
    W0, H0 = level_dims(slide, 0)

    thumb_w = max(1, W0 // max(1, downsample))
    thumb_h = max(1, H0 // max(1, downsample))
    thumb = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    draw = ImageDraw.Draw(thumb)

    sx = thumb.width / float(W0)
    sy = thumb.height / float(H0)

    pw0, ph0 = tile_size_level0(slide, sample_level, patch_size)
    for (x0, y0) in coors_lvl0:
        x1, y1 = x0 + pw0, y0 + ph0
        draw.rectangle([int(x0 * sx), int(y0 * sy), int(x1 * sx), int(y1 * sy)],
                       outline=(255, 0, 0), width=1)
    return thumb


# =========================
# Otsu mask and integral image
# =========================
def get_otsu_mask(
    slide_path: str,
    bg_level: int = 2,
    kernel: int = 5,
    open_iter: int = 1,
    close_iter: int = 1,
    min_area: int = 500,
    keep_largest: bool = False,
    invert: bool = False,
    fallback_max_edge: int = 4096,
    gaussian_ksize: int = 0,
) -> np.ndarray:
    img = None
    try:
        with openslide.open_slide(slide_path) as s:
            lvl = int(bg_level)
            lvl = max(0, min(lvl, s.level_count - 1))
            W, H = s.level_dimensions[lvl]
            img = np.array(s.read_region((0, 0), lvl, (W, H)).convert("RGB"))
    except Exception:
        try:
            with openslide.open_slide(slide_path) as s2:
                thumb = s2.get_thumbnail((fallback_max_edge, fallback_max_edge)).convert("RGB")
                img = np.array(thumb)
        except Exception as ee:
            raise ee

    max_edge_soft = 8192
    h, w = img.shape[:2]
    if max(h, w) > max_edge_soft:
        scale = max_edge_soft / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if gaussian_ksize and gaussian_ksize >= 3 and gaussian_ksize % 2 == 1:
        gray = cv2.GaussianBlur(gray, (gaussian_ksize, gaussian_ksize), 0)
    _, binimg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask = (binimg == 0).astype(np.uint8)
    if invert:
        mask = 1 - mask

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(1, kernel), max(1, kernel)))
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=int(open_iter))
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=int(close_iter))

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8, ltype=cv2.CV_32S)
    if keep_largest:
        if num_labels <= 1:
            kept = np.zeros_like(mask, dtype=np.uint8)
        else:
            largest_id = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            kept = (labels == largest_id).astype(np.uint8)
    else:
        areas = stats[:, cv2.CC_STAT_AREA]
        lut = np.zeros(num_labels, dtype=np.uint8)
        lut[areas >= int(min_area)] = 1
        lut[0] = 0
        kept = lut[labels]
    return kept.astype(np.uint8)


def integral_image(mask01: np.ndarray) -> np.ndarray:
    h, w = mask01.shape
    S = np.zeros((h + 1, w + 1), dtype=np.int64)
    S[1:, 1:] = mask01.astype(np.int64)
    S = S.cumsum(axis=0)
    S = S.cumsum(axis=1)
    return S


def region_sum(S: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    x2, y2 = x + w, y + h
    return int(S[y2, x2] - S[y, x2] - S[y2, x] + S[y, x])


# =========================
# Sampling
# =========================
def grid_sample_with_otsu(slide_path: str,
                          sample_level: int,
                          patch_size: int,
                          stride: int,
                          mask_cfg: dict,
                          min_fg_ratio: float,
                          use_bg_mask: bool = True) -> List[Tuple[int, int]]:
    slide = openslide.open_slide(slide_path)
    Ws, Hs = level_dims(slide, sample_level)
    if Ws < patch_size or Hs < patch_size:
        return []

    sx_s20, sy_s20 = scale_sample_to_level0(slide, sample_level)
    pw0, ph0 = tile_size_level0(slide, sample_level, patch_size)
    W0, H0 = level_dims(slide, 0)

    S_low = None
    if use_bg_mask:
        fg_low = get_otsu_mask(
            slide_path,
            bg_level=int(mask_cfg.get("bg_mask_level", 2)),
            kernel=int(mask_cfg.get("kernel", 5)),
            open_iter=int(mask_cfg.get("open_iter", 1)),
            close_iter=int(mask_cfg.get("close_iter", 1)),
            min_area=int(mask_cfg.get("min_area", 500)),
            keep_largest=bool(mask_cfg.get("keep_largest", False)),
            invert=bool(mask_cfg.get("invert", False)),
            fallback_max_edge=int(mask_cfg.get("fallback_max_edge", 4096)),
            gaussian_ksize=int(mask_cfg.get("gaussian_ksize", 0)),
        )
        S_low = integral_image(fg_low)
        H_low, W_low = fg_low.shape
        sx_s2l = W_low / float(Ws)
        sy_s2l = H_low / float(Hs)
    else:
        sx_s2l = sy_s2l = None

    nx = int(math.ceil((Ws - patch_size) / float(stride))) + 1
    ny = int(math.ceil((Hs - patch_size) / float(stride))) + 1
    max_x_s = max(0, Ws - patch_size)
    max_y_s = max(0, Hs - patch_size)

    coors_lvl0, seen = [], set()
    for j in range(ny):
        y_s = min(j * stride, max_y_s)
        for i in range(nx):
            x_s = min(i * stride, max_x_s)

            if S_low is not None:
                xL = int(x_s * sx_s2l); yL = int(y_s * sy_s2l)
                wL = max(1, int(math.ceil(patch_size * sx_s2l)))
                hL = max(1, int(math.ceil(patch_size * sy_s2l)))
                if xL < 0 or yL < 0 or (xL + wL) > W_low or (yL + hL) > H_low:
                    continue
                if region_sum(S_low, xL, yL, wL, hL) < int(min_fg_ratio * wL * hL):
                    continue

            x0 = min(int(round(x_s * sx_s20)), max(0, W0 - pw0))
            y0 = min(int(round(y_s * sy_s20)), max(0, H0 - ph0))
            key = (x0, y0)
            if key not in seen:
                seen.add(key)
                coors_lvl0.append(key)
    return coors_lvl0


def random_sample_with_otsu(slide_path: str,
                            sample_level: int,
                            patch_size: int,
                            n_tiles: int,
                            seed: int,
                            max_trials: int,
                            mask_cfg: dict,
                            min_fg_ratio: float = 0.5,
                            use_bg_mask: bool = True) -> List[Tuple[int, int]]:
    slide = openslide.open_slide(slide_path)
    Ws, Hs = level_dims(slide, sample_level)
    if Ws < patch_size or Hs < patch_size or n_tiles <= 0:
        return []

    sx_s20, sy_s20 = scale_sample_to_level0(slide, sample_level)
    pw0, ph0 = tile_size_level0(slide, sample_level, patch_size)
    W0, H0 = level_dims(slide, 0)

    S_low = None
    if use_bg_mask:
        fg_low = get_otsu_mask(
            slide_path,
            bg_level=int(mask_cfg.get("bg_mask_level", 2)),
            kernel=int(mask_cfg.get("kernel", 5)),
            open_iter=int(mask_cfg.get("open_iter", 1)),
            close_iter=int(mask_cfg.get("close_iter", 1)),
            min_area=int(mask_cfg.get("min_area", 500)),
            keep_largest=bool(mask_cfg.get("keep_largest", False)),
            invert=bool(mask_cfg.get("invert", False)),
            fallback_max_edge=int(mask_cfg.get("fallback_max_edge", 4096)),
            gaussian_ksize=int(mask_cfg.get("gaussian_ksize", 0)),
        )
        S_low = integral_image(fg_low)
        H_low, W_low = fg_low.shape
        sx_s2l = W_low / float(Ws)
        sy_s2l = H_low / float(Hs)
    else:
        sx_s2l = sy_s2l = None

    rng = random.Random(seed)
    x_max_s = Ws - patch_size
    y_max_s = Hs - patch_size

    picked, seen, trials = [], set(), 0
    while len(picked) < n_tiles and trials < max_trials:
        trials += 1
        x_s = rng.randint(0, x_max_s)
        y_s = rng.randint(0, y_max_s)

        if S_low is not None:
            xL = int(x_s * sx_s2l); yL = int(y_s * sy_s2l)
            wL = max(1, int(math.ceil(patch_size * sx_s2l)))
            hL = max(1, int(math.ceil(patch_size * sy_s2l)))
            if xL < 0 or yL < 0 or (xL + wL) > W_low or (yL + hL) > H_low:
                continue
            if region_sum(S_low, xL, yL, wL, hL) < int(min_fg_ratio * wL * hL):
                continue

        x0 = min(int(round(x_s * sx_s20)), max(0, W0 - pw0))
        y0 = min(int(round(y_s * sy_s20)), max(0, H0 - ph0))
        key = (x0, y0)
        if key not in seen:
            seen.add(key)
            picked.append(key)
    return picked


def save_tissue_mask_preview(
    slide_path: str,
    sample_level: int,
    mask_cfg: dict,
    out_path: str,
    max_edge: int = 4096
):
    fg_low = get_otsu_mask(
        slide_path,
        bg_level=int(mask_cfg.get("bg_mask_level", 2)),
        kernel=int(mask_cfg.get("kernel", 5)),
        open_iter=int(mask_cfg.get("open_iter", 1)),
        close_iter=int(mask_cfg.get("close_iter", 1)),
        min_area=int(mask_cfg.get("min_area", 500)),
        keep_largest=bool(mask_cfg.get("keep_largest", False)),
        invert=bool(mask_cfg.get("invert", False)),
        fallback_max_edge=int(mask_cfg.get("fallback_max_edge", 4096)),
        gaussian_ksize=int(mask_cfg.get("gaussian_ksize", 0)),
    )

    with openslide.open_slide(slide_path) as s2:
        thumb = s2.get_thumbnail((max_edge, max_edge)).convert("RGB")
    tw, th = thumb.size
    thumb_np = np.array(thumb)

    mask_vis = cv2.resize(fg_low, (tw, th), interpolation=cv2.INTER_NEAREST)
    gray = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2GRAY)
    gray3 = np.stack([gray, gray, gray], axis=-1)
    overlay = gray3.copy()
    overlay[mask_vis == 1] = (0.6 * overlay[mask_vis == 1] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)

    Image.fromarray(overlay).save(out_path, quality=90)
    log(f"Tissue-mask preview saved to {out_path} (preview={tw}x{th}, mask_src={fg_low.shape[::-1]})")


def safe_read_region(slide, loc, level, size, retries=2, sleep=0.01):
    x, y = loc; w, h = size
    for t in range(retries + 1):
        try:
            return slide.read_region((x, y), level, (w, h)).convert("RGB")
        except Exception:
            if t == retries:
                return None
            time.sleep(sleep)
    return None


# =========================
# Feature extraction
# =========================
class ResNetFeature(nn.Module):
    def __init__(self, backbone: str = "resnet34", pooling: bool = True, pretrained: bool = True):
        super().__init__()
        name = backbone.lower()
        valid = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
        assert name in valid, f"backbone must be one of {valid}"

        weights = None
        if pretrained:
            weight_map = {
                "resnet18": torchvision.models.ResNet18_Weights.DEFAULT,
                "resnet34": torchvision.models.ResNet34_Weights.DEFAULT,
                "resnet50": torchvision.models.ResNet50_Weights.DEFAULT,
                "resnet101": torchvision.models.ResNet101_Weights.DEFAULT,
                "resnet152": torchvision.models.ResNet152_Weights.DEFAULT,
            }
            weights = weight_map[name]

        base_model = getattr(torchvision.models, name)(weights=weights)
        self.pooling = pooling
        self.out_dim = 512 if name in ["resnet18", "resnet34"] else 2048
        self.features = nn.Sequential(*list(base_model.children())[:-2])

    def forward(self, x):
        x = self.features(x)
        if self.pooling:
            x = x.view(x.size(0), x.size(1), -1).mean(dim=-1)
        return x


class PatchDataset(Dataset):
    def __init__(self, slide_path: str, coors_level0: List[Tuple[int, int]],
                 patch_size: int, sample_level: int):
        self.slide_path = slide_path
        self.coors = coors_level0
        self.patch_size = patch_size
        self.sample_level = sample_level
        self._slide = None

        self.tf = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _ensure_open(self):
        if self._slide is None:
            self._slide = openslide.open_slide(self.slide_path)

    def __len__(self):
        return len(self.coors)

    def __getitem__(self, idx: int):
        self._ensure_open()
        x0, y0 = self.coors[idx]
        img = safe_read_region(self._slide, (x0, y0), self.sample_level,
                               (self.patch_size, self.patch_size),
                               retries=2, sleep=0.01)
        if img is None:
            # Replace a failed patch read with a black tile so the pipeline can continue.
            img = Image.fromarray(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.uint8))
        return self.tf(img)


def infer_backbone_out_dim(backbone: str) -> int:
    name = (backbone or "resnet34").lower()
    return 512 if name in ["resnet18", "resnet34"] else 2048


def extract_features(slide_path: str,
                     coors_level0: List[Tuple[int, int]],
                     patch_size: int,
                     sample_level: int,
                     model_cfg: dict) -> torch.Tensor:
    if len(coors_level0) == 0:
        # Return an empty feature tensor with the expected backbone dimension.
        out_dim = infer_backbone_out_dim(model_cfg.get("backbone", "resnet34"))
        return torch.zeros((0, out_dim), dtype=torch.float32)

    device = model_cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = ResNetFeature(
        backbone=model_cfg.get("backbone", "resnet34"),
        pooling=bool(model_cfg.get("pooling", True)),
        pretrained=bool(model_cfg.get("pretrained", True))
    ).to(device)
    model.eval()

    loader_cfg = model_cfg.get("loader", {})
    base_batch = int(loader_cfg.get("batch_size", 1024))
    default_workers = 0 if os.name == "nt" else 4
    base_workers = int(loader_cfg.get("num_workers", default_workers))
    pin_mem_cfg = bool(loader_cfg.get("pin_memory", True))

    def build_loader(num_workers: int, pin_memory: bool, batch_size: int):
        ds = PatchDataset(slide_path, coors_level0, patch_size, sample_level)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0 and bool(loader_cfg.get("persistent_workers", False))),
        )

    def run(num_workers: int, pin_memory: bool, batch_size: int, tag: str):
        loader = build_loader(num_workers, pin_memory, batch_size)
        outs = []
        try:
            with torch.no_grad():
                for batch in tqdm(loader, desc=f"Extracting[{tag}]", unit="batch", dynamic_ncols=True):
                    batch = batch.to(device, non_blocking=(pin_memory and device == "cuda"))
                    out = model(batch)
                    outs.append(out.detach().cpu())
            return torch.cat(outs, dim=0)
        finally:
            try:
                it = getattr(loader, "_iterator", None)
                if it is not None:
                    it._shutdown_workers()
            except Exception:
                pass
            del loader
            import gc
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    trials = [
        (base_workers, pin_mem_cfg and device.startswith("cuda"), base_batch,
         f"wkr={base_workers},pin={pin_mem_cfg and device.startswith('cuda')},b={base_batch}"),
        (base_workers, False, base_batch, f"wkr={base_workers},pin=False,b={base_batch}"),
        (0, False, base_batch, f"wkr=0,pin=False,b={base_batch}"),
        (0, False, max(1, base_batch // 2), f"wkr=0,pin=False,b={max(1, base_batch // 2)}"),
    ]

    last_err = None
    for (nw, pm, bs, tag) in trials:
        try:
            return run(nw, pm, bs, tag)
        except Exception as e:
            last_err = e
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            log(f"[WARN] Feature-extraction configuration failed ({tag}): {e}")

    raise last_err


# =========================
# Sampling worker
# =========================
def sampling_only_worker(args: Tuple[str, dict, dict, dict, str]) -> Dict:
    slide_path, sp, mask_cfg, vis, out_dirs = args
    patch_coors_dir, sampled_vis_dir = out_dirs.split("|")
    slide_id = slide_id_from_path(slide_path)

    # Reuse readable coordinate files to support resume/restart.
    coor_file = os.path.join(patch_coors_dir, f"{slide_id}_coors.npy")
    if os.path.exists(coor_file):
        arr = safe_load_npy(coor_file)
        if arr is not None:
            return {"slide_id": slide_id, "ok": True, "n_patches": int(arr.shape[0]), "coor_file": coor_file, "from_cache": True}

    # A) Verify that the slide can be opened.
    try:
        with suppress_c_stderr():
            s = openslide.open_slide(slide_path)
            s.close()
    except Exception as e:
        return {"slide_id": slide_id, "ok": False, "err": f"Failed to open slide: {e}"}

    use_bg_mask = bool(sp.get("use_bg_mask", True))
    min_fg_ratio = float(sp.get("min_fg_ratio", 0.5))
    mode = sp.get("mode", "dense").lower()
    sample_level = int(sp.get("sample_level", 0))
    patch_size = int(sp.get("patch_size", 256))

    # B) Sample coordinates.
    with suppress_c_stderr():
        if mode == "dense":
            stride = int(sp.get("dense_stride", patch_size))
            sid = short_id(slide_id)
            log(f"[{sid}] Dense + OtsuMask, level={sample_level}, patch={patch_size}, stride={stride}, min_fg_ratio={min_fg_ratio}")
            coors_lvl0 = grid_sample_with_otsu(
                slide_path, sample_level, patch_size, stride,
                mask_cfg, min_fg_ratio=min_fg_ratio, use_bg_mask=use_bg_mask
            )
        elif mode == "random":
            num_sample = int(sp.get("num_sample", 2000))
            seed = int(sp.get("seed", 7))
            max_trials = int(sp.get("max_trials", 100000))
            sid = short_id(slide_id)
            log(f"[{sid}] Random + OtsuMask, level={sample_level}, patch={patch_size}, n={num_sample}, seed={seed}, min_fg_ratio={min_fg_ratio}")
            coors_lvl0 = random_sample_with_otsu(
                slide_path, sample_level, patch_size, num_sample, seed, max_trials,
                mask_cfg, min_fg_ratio=min_fg_ratio, use_bg_mask=use_bg_mask
            )
        else:
            return {"slide_id": slide_id, "ok": False, "err": f"Unknown sampling mode: {mode}"}

    # C) Save coordinates atomically.
    try:
        atomic_save_npy(np.array(coors_lvl0, dtype=np.int32), coor_file)
    except Exception as e:
        return {"slide_id": slide_id, "ok": False, "err": f"Failed to save coordinates: {e}"}

    # D) Save optional previews and overlays.
    if bool(mask_cfg.get("preview", False)):
        preview_out = os.path.join(sampled_vis_dir, f"{slide_id}_tissuemask_preview.jpg")
        if not os.path.exists(preview_out):
            try:
                with suppress_c_stderr():
                    save_tissue_mask_preview(slide_path, sample_level, mask_cfg, preview_out)
            except Exception as e:
                sid = short_id(slide_id)
                log(f"[{sid}] Tissue-mask preview failed and was skipped: {e}")

    if bool(vis.get("enable", True)):
        downsample = int(vis.get("downsample", 32))
        vis_out = os.path.join(sampled_vis_dir, f"{slide_id}_sampled_patches.jpg")
        if not os.path.exists(vis_out):
            try:
                with suppress_c_stderr():
                    thumb = draw_patches_on_slide_level0(slide_path, coors_lvl0, patch_size, sample_level, downsample)
                os.makedirs(sampled_vis_dir, exist_ok=True)
                thumb.save(vis_out, quality=90)
            except Exception as e:
                sid = short_id(slide_id)
                log(f"[{sid}] Sampling visualization failed and was skipped: {e}")

    return {"slide_id": slide_id, "ok": True, "n_patches": len(coors_lvl0), "coor_file": coor_file, "from_cache": False}


# =========================
# GPU consumer thread
# =========================
def feature_consumer(name: str,
                     task_q: "queue.Queue[Tuple[str,str,str]]",
                     save_dir: str,
                     sample_level: int,
                     patch_size: int,
                     model_cfg: dict,
                     stop_event: threading.Event):
    patch_ft_dir = os.path.join(save_dir, "patch_ft")
    while not stop_event.is_set():
        try:
            item = task_q.get(timeout=1.0)
        except queue.Empty:
            continue
        if item is None:
            task_q.task_done()
            break

        slide_id, slide_path, coor_file = item
        try:
            # Skip already completed features to support resume.
            fts_path = os.path.join(patch_ft_dir, f"{slide_id}_fts.npy")
            if os.path.exists(fts_path) and safe_load_npy(fts_path) is not None:
                log(f"[{name}] Skipping completed features: {slide_id}")
                task_q.task_done()
                continue

            # Broken coordinates are skipped so sampling can be recomputed upstream.
            coors = safe_load_npy(coor_file)
            if coors is None:
                log(f"[{name}] Coordinates are missing or broken, skipping: {coor_file}")
                task_q.task_done()
                continue

            coors_list = coors.tolist()
            log(f"[{name}] Starting feature extraction: {slide_id} (patches={len(coors_list)})")
            feats = extract_features(slide_path, coors_list, patch_size, sample_level, model_cfg)
            atomic_save_npy(feats.numpy(), fts_path)
            log(f"[{name}] Features saved: {fts_path}, shape={tuple(feats.shape)}")

        except Exception:
            traceback.print_exc()
            log(f"[{name}] Feature extraction failed and the slide was skipped: {slide_id}")
        finally:
            task_q.task_done()


# =========================
# Main pipeline
# =========================
def main(cfg: dict):
    save_dir = cfg["save_dir"]
    patch_ft_dir = os.path.join(save_dir, "patch_ft")
    patch_coors_dir = os.path.join(save_dir, "patch_coor")
    sampled_vis_dir = os.path.join(save_dir, "sampled_vis")
    for d in [save_dir, patch_ft_dir, patch_coors_dir, sampled_vis_dir]:
        check_dir(d)

    if "seed" in cfg.get("sampling", {}):
        set_seed(int(cfg["sampling"]["seed"]))

    sp = cfg["sampling"]
    mask_cfg = cfg.get("mask", {})
    vis = cfg.get("visualization", {})
    model_cfg = cfg.get("model", {})
    rt = cfg.get("runtime", {})

    exts = tuple(cfg.get("exts", (".svs", ".tif", ".ndpi")))
    all_wsi = build_wsi_paths_from_root(cfg["data_root"], exts=exts)

    target_files = rt.get("target_files", []) or []
    if len(target_files) > 0:
        log(f"[Filter] Found {len(target_files)} filenames in target_files. Applying whitelist filtering.")
        target_set = set(target_files)
        all_wsi = [p for p in all_wsi if os.path.basename(p) in target_set]
        log(f"[Filter] Filtering finished. Slides to process: {len(all_wsi)}")
        if not all_wsi:
            log("[Warning] None of the requested filenames were found under data_root.")
    else:
        log("[Filter] target_files is empty. Processing all supported slides under data_root.")


    if not all_wsi:
        log(f"No supported slide files were found under {cfg['data_root']}: {exts}")
        return

    cpu_workers = int(rt.get("cpu_workers", os.cpu_count() or 1))
    gpu_workers = int(os.environ.get("HYPERMIL_GPU_WORKERS", rt.get("gpu_workers", 1)))
    q_maxsize = int(rt.get("queue_maxsize", 8))
    gpu_ids = rt.get("gpu_ids", None)  # Optional, e.g. [0, 1]
    gpu_ids_env = os.environ.get("HYPERMIL_GPU_IDS", "").strip()
    if gpu_ids_env:
        gpu_ids = [int(x.strip()) for x in gpu_ids_env.split(",") if x.strip()]
    elif gpu_ids is None:
        gpu_ids = []
    elif isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    else:
        gpu_ids = [int(x) for x in gpu_ids if str(x).strip() != ""]

    # Split slides into completed, ready-for-feature, and sampling-required groups.
    done_ids = list_done_feature_ids(patch_ft_dir)
    ready_for_feat, need_sampling = [], []
    for spath in all_wsi:
        sid = slide_id_from_path(spath)
        if sid in done_ids:
            continue
        coor_file = os.path.join(patch_coors_dir, f"{sid}_coors.npy")
        if os.path.exists(coor_file) and safe_load_npy(coor_file) is not None:
            ready_for_feat.append((sid, spath, coor_file))
        else:
            need_sampling.append(spath)

    log(
        f"Slides={len(all_wsi)}; completed_features={len(done_ids)}; "
        f"ready_for_features={len(ready_for_feat)}; need_sampling={len(need_sampling)}"
    )

    # Limit BLAS thread fan-out inside worker processes.
    env_vars = ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]
    backups = {k: os.environ.get(k) for k in env_vars}
    for k in env_vars:
        os.environ[k] = "1"

    # Build the GPU task queue and consumer threads.
    task_q: "queue.Queue[Tuple[str,str,str]]" = queue.Queue(maxsize=q_maxsize)
    stop_event = threading.Event()
    consumers: List[threading.Thread] = []

    def build_model_cfg_for_worker(worker_idx: int) -> dict:
        mc = dict(model_cfg)
        if gpu_ids is not None and len(gpu_ids) > 0:
            gid = gpu_ids[worker_idx % len(gpu_ids)]
            mc["device"] = f"cuda:{gid}"
        return mc

    for w in range(max(1, gpu_workers)):
        mc = build_model_cfg_for_worker(w)
        th = threading.Thread(
            target=feature_consumer,
            args=(f"GPUWorker-{w}", task_q, save_dir,
                  int(sp.get("sample_level", 0)), int(sp.get("patch_size", 256)),
                  mc, stop_event),
            daemon=True
        )
        th.start()
        consumers.append(th)

    # Feed slides with precomputed coordinates into the queue in the background.
    def _feed_ready_async():
        fed = 0
        total = len(ready_for_feat)
        for (sid, spath, cfile) in ready_for_feat:
            while not stop_event.is_set():
                try:
                    task_q.put((sid, spath, cfile), timeout=0.1)
                    fed += 1
                    if fed % 50 == 0 or fed == total:
                        log(f"[feeder] queued ready-made coordinates: {fed}/{total}")
                    break
                except queue.Full:
                    time.sleep(0.05)
        log(f"[feeder] finished queuing ready-made coordinates: {fed}/{total}")

    feeder = threading.Thread(target=_feed_ready_async, name="ReadyFeeder", daemon=True)
    feeder.start()

    # Submit sampling jobs while the feature queue is consumed in parallel.
    args_list = []
    for slide_path in need_sampling:
        args_list.append((
            slide_path, sp, mask_cfg, vis,
            f"{patch_coors_dir}|{sampled_vis_dir}"
        ))

    ok_cnt = 0
    try:
        with ProcessPoolExecutor(max_workers=cpu_workers) as ex:
            fut2sp = {ex.submit(sampling_only_worker, a): a[0] for a in args_list}
            for fut in as_completed(fut2sp):
                spath = fut2sp[fut]
                sid = slide_id_from_path(spath)
                try:
                    res = fut.result()
                    if res.get("ok", False):
                        ok_cnt += 1
                        coor_file = res.get("coor_file")
                        # Queue each slide as soon as sampling finishes.
                        while not stop_event.is_set():
                            try:
                                task_q.put((sid, spath, coor_file), timeout=0.1)
                                break
                            except queue.Full:
                                time.sleep(0.05)
                        log(f"[sampling] finished {sid} patches={res.get('n_patches', 0)} [{ok_cnt}/{len(args_list)}]")
                    else:
                        log(f"[sampling] failed {sid} err={res.get('err')}")
                except Exception as e:
                    log(f"[sampling] failed {sid} err={e}")
    finally:
        # Wait for the background feeder to finish.
        feeder.join()

        # Send termination sentinels to GPU workers.
        for _ in consumers:
            task_q.put(None)
        task_q.join()
        stop_event.set()
        for th in consumers:
            th.join()

        # Restore environment variables.
        for k, v in backups.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    log("All preprocessing tasks finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH, help="Path to the YAML config file.")
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # Flatten the config layout for backward compatibility.
    cfg = {
        "data_root": cfg.get("data_root"),
        "save_dir": cfg.get("save_dir"),
        "sampling": cfg.get("sampling", {}),
        "model": cfg.get("model", {}),
        "visualization": cfg.get("visualization", {}),
        "mask": cfg.get("mask", {}),
        "runtime": cfg.get("runtime", {}),
        "exts": cfg.get("exts", [".svs", ".tif", ".ndpi"]),
    }

    for key in ("data_root", "save_dir"):
        value = cfg.get(key)
        if value and not os.path.isabs(value):
            cfg[key] = os.path.abspath(os.path.join(PROJECT_ROOT, value))
    main(cfg)
