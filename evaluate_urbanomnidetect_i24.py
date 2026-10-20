#!/usr/bin/env python3
"""
Evaluate UrbanOmniDetect on I24-3D cached frames.

Expected I24 cache layout (created by I24-3D-dataset/export_dataset.py):
  CACHE/im/scene_1/p1c1/0000.png
  CACHE/label/scene_1/p1c1/0000.cpkl

The .cpkl labels must contain `im_box`, an 8x2 projection of the I24 3D box.
This script performs class-agnostic vehicle evaluation: all I24 objects and all
UrbanOmniDetect predictions are assigned class 0. This avoids incompatible
I24 vehicle subclasses vs COCO class IDs. It reports AP2D/APBEV/AP3D using
UrbanOmniDetect's evaluator functions.

Run from the UrbanOmniDetect repository root, with the I24-3D repository on
PYTHONPATH. For v1 checkpoints, GT I24 corners are reordered from
(bottom four, top four) to UrbanOmniDetect v1's (top four, bottom four).
For v2 checkpoints, no half-swap is applied.
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True,
                        help="I24 cache root containing im/ and label/")
    parser.add_argument("--scene", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--camera", required=True, help="e.g. p1c1")
    parser.add_argument("--model", required=True, help="UrbanOmniDetect .pt checkpoint")
    parser.add_argument("--uod-version", choices=["v1", "v2"], default="v1")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", default="0.25,0.5,0.7")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--ground-indices", default=None,
                        help="Optional forced ground keypoint indices, e.g. 4,5,6,7")
    parser.add_argument("--report-csv", default="i24_urbanomnidetect_eval.csv")
    args = parser.parse_args()

    # Import repository evaluator helpers. Run this script from the UOD repo root.
    sys.path.insert(0, os.getcwd())
    try:
        from eval3d import (
            GTBox, compute_ap, resolve_ground, solve_image_homographies,
            _ordered_polygon, iou_2d, iou_bev, iou_3d, _HAVE_SHAPELY
        )
        from homography_rt import OrthoHomographySolver
        from uod.keypoints import parse_pose_result
        if args.uod_version == "v2":
            import v2.hybrid_v2  # noqa: F401; registers custom model class
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Could not import UrbanOmniDetect code/dependencies. Run from the "
            "urbanomnidetect repository root and install requirements.txt."
        ) from exc

    if not _HAVE_SHAPELY:
        raise RuntimeError("Install shapely; UrbanOmniDetect BEV/3D evaluation needs it.")

    image_dir = Path(args.cache) / "im" / f"scene_{args.scene}" / args.camera
    label_dir = Path(args.cache) / "label" / f"scene_{args.scene}" / args.camera
    images = sorted(glob.glob(str(image_dir / "*.png")))
    if args.max_images:
        images = images[:args.max_images]
    if not images:
        raise FileNotFoundError(f"No .png images found in {image_dir}")

    forced = [int(x) for x in args.ground_indices.split(",")] if args.ground_indices else None
    model = YOLO(args.model)

    preds_all, gts_all, raw_kpts = [], [], []
    evaluated_images = []

    for image_path in images:
        basename = Path(image_path).stem
        label_path = label_dir / f"{basename}.cpkl"
        if not label_path.exists():
            # Do not count frames without corresponding annotations.
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"WARNING: unreadable image, skipping: {image_path}")
            continue
        height, width = image.shape[:2]

        with open(label_path, "rb") as f:
            objects = pickle.load(f)

        gt_boxes = []
        for obj in objects:
            if "im_box" not in obj:
                continue
            kpts = obj["im_box"]
            if torch.is_tensor(kpts):
                kpts = kpts.detach().cpu().numpy()
            kpts = np.asarray(kpts, dtype=np.float64).reshape(-1, 2)
            if kpts.shape != (8, 2) or not np.isfinite(kpts).all():
                continue

            # I24 state_to_im convention is bottom/ground four followed by top four.
            # UrbanOmniDetect v1 convention is top four followed by ground four.
            if args.uod_version == "v1":
                kpts = np.concatenate([kpts[4:8], kpts[0:4]], axis=0)

            # Exclude boxes whose projected corners are all outside the image.
            if (kpts[:, 0].max() < 0 or kpts[:, 1].max() < 0 or
                    kpts[:, 0].min() >= width or kpts[:, 1].min() >= height):
                continue

            # Clip only the rectangle to image bounds; preserve keypoints for geometry.
            x1 = float(np.clip(kpts[:, 0].min(), 0, width - 1))
            y1 = float(np.clip(kpts[:, 1].min(), 0, height - 1))
            x2 = float(np.clip(kpts[:, 0].max(), 0, width - 1))
            y2 = float(np.clip(kpts[:, 1].max(), 0, height - 1))
            gt_boxes.append(GTBox(cls=0, kpts=kpts, xyxy=np.array([x1, y1, x2, y2]), conf=1.0))

        result = model.predict(
            image, imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False
        )[0]

        detections = []
        if result.keypoints is not None and result.keypoints.data is not None:
            raw = result.keypoints.data.detach().cpu().numpy()
            if len(raw):
                # Keep XY only; keypoint confidence is not used by eval3d.py.
                raw_xy = raw[:, :, :2]
                raw_kpts.extend(list(raw_xy))
                # parse_pose_result returns the detector's box, class and confidence.
                parsed = parse_pose_result(result, list(range(8)))
                for det in parsed:
                    if len(det.kpts) < 8:
                        continue
                    kpts = np.asarray(det.kpts[:8], dtype=np.float64)
                    # Class-agnostic vehicle evaluation: predictions and GT class=0.
                    detections.append(GTBox(
                        cls=0,
                        kpts=kpts,
                        xyxy=np.asarray(det.xyxy, dtype=np.float64),
                        conf=float(det.conf),
                    ))

        preds_all.append(detections)
        gts_all.append(gt_boxes)
        evaluated_images.append(image_path)

    if not evaluated_images:
        raise RuntimeError(
            f"No paired image/label files were evaluated. Expected matching .cpkl "
            f"files in {label_dir}"
        )

    ground_idx = resolve_ground(raw_kpts, forced)
    print(f"Evaluated images: {len(evaluated_images)}")
    print(f"Ground keypoint indices: {ground_idx}")
    print("Evaluation is class-agnostic; all detected objects are treated as vehicles.")

    solver = OrthoHomographySolver(warm_start=False)
    homographies = solve_image_homographies(preds_all, gts_all, ground_idx, solver)

    thresholds = [float(x.strip()) for x in args.iou.split(",") if x.strip()]
    rows = []
    print(f"\n{'IoU':>6} {'AP2D':>10} {'APBEV':>10} {'AP3D':>10}")
    for threshold in thresholds:
        ap2d = compute_ap(preds_all, gts_all, threshold, "2d", ground_idx)
        apbev = compute_ap(preds_all, gts_all, threshold, "bev", ground_idx, homographies)
        ap3d = compute_ap(preds_all, gts_all, threshold, "3d", ground_idx, homographies)
        print(f"{threshold:6.2f} {ap2d:10.4f} {apbev:10.4f} {ap3d:10.4f}")
        rows.append((threshold, ap2d, apbev, ap3d))

    import csv
    with open(args.report_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["iou_threshold", "AP2D_11pt", "APBEV_11pt", "AP3D_11pt"])
        writer.writerows(rows)
    print(f"\nSaved results to: {args.report_csv}")
    print("Note: APBEV/AP3D are homography-based estimates, not direct 3D XYZ IoU.")


if __name__ == "__main__":
    main()
