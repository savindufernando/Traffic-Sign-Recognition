"""
Crop sign-likely ROI regions from extracted dashcam frames.

Takes the frames extracted by extract_dashcam_frames.py and crops
multiple candidate regions per frame where signs typically appear.
The crops are then classified by the EXISTING model, and only those
with confidence > threshold are kept as real-world training data.

This is a "semi-automatic annotation" approach:
1. Crop 6 ROI zones from each frame
2. Run existing classifier on each crop
3. Keep crops where confidence > threshold (likely real signs)
4. Save to class-organised folders for fine-tuning

Output goes to a NEW directory — never touches existing data.

Usage:
    python scripts/crop_sign_regions.py \
        --frames ./dashcam_frames \
        --output ./dashcam_crops \
        --threshold 0.3
"""

import sys
import argparse
import shutil
from pathlib import Path
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_roi_regions(w: int, h: int):
    """Get candidate ROI crop boxes where signs typically appear."""
    return [
        (0,           int(h*0.10), int(w*0.40), int(h*0.60)),   # Left roadside
        (int(w*0.60), int(h*0.10), w,           int(h*0.60)),   # Right roadside
        (int(w*0.20), int(h*0.05), int(w*0.80), int(h*0.45)),   # Center-top
        (0,           int(h*0.20), int(w*0.30), int(h*0.70)),   # Left close
        (int(w*0.70), int(h*0.20), w,           int(h*0.70)),   # Right close
        (int(w*0.15), int(h*0.15), int(w*0.85), int(h*0.65)),   # Wide center
    ]


def main():
    parser = argparse.ArgumentParser(description="Crop sign regions from dashcam frames")
    parser.add_argument("--frames", "-f", type=str, default="./dashcam_frames",
                        help="Directory of extracted frames")
    parser.add_argument("--output", "-o", type=str, default="./dashcam_crops",
                        help="Output directory for classified crops")
    parser.add_argument("--threshold", "-t", type=float, default=0.3,
                        help="Min confidence to keep a crop (default: 0.3 — intentionally low to gather more data)")
    parser.add_argument("--max-frames", "-m", type=int, default=None,
                        help="Max frames to process (for testing)")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output_dir = Path(args.output)

    if not frames_dir.exists():
        print(f"Error: Frames directory not found: {frames_dir}")
        print(f"Run extract_dashcam_frames.py first!")
        sys.exit(1)

    # Load the existing model (the original, never modified)
    print("Loading existing TSR model (read-only, not modifying it)...")
    from src.inference.predictor import load_predictor

    model_path = Path(__file__).parent.parent / "models_sri_lanka" / "best_model.pth"
    config_path = Path(__file__).parent.parent / "config_sri_lanka.yaml"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)

    predictor = load_predictor(str(model_path), str(config_path))
    print(f"Model loaded successfully.\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    frames = sorted(frames_dir.glob("*.jpg"))
    if args.max_frames:
        frames = frames[:args.max_frames]

    print(f"Processing {len(frames)} frames...")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Output: {output_dir}\n")

    total_kept = 0
    class_counts = {}

    for i, frame_path in enumerate(frames):
        img = Image.open(frame_path).convert("RGB")
        w, h = img.size
        rois = get_roi_regions(w, h)

        for j, roi_box in enumerate(rois):
            crop = img.crop(roi_box)
            result = predictor.predict(crop, return_all_probs=False)

            if result["confidence"] >= args.threshold and result["is_confident"]:
                class_name = result["class_name"]
                class_dir = output_dir / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                crop_filename = f"{frame_path.stem}_roi{j}.jpg"
                crop.save(str(class_dir / crop_filename))

                class_counts[class_name] = class_counts.get(class_name, 0) + 1
                total_kept += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(frames)} frames ({total_kept} crops kept)")

    # Summary
    print(f"\n{'='*50}")
    print(f"Done! Kept {total_kept} crops across {len(class_counts)} classes")
    print(f"Output: {output_dir}")
    print(f"\nTop classes:")
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {cls}: {count}")
    print(f"{'='*50}")

    # Save stats
    import json
    stats = {
        "total_frames_processed": len(frames),
        "total_crops_kept": total_kept,
        "threshold": args.threshold,
        "class_counts": class_counts
    }
    with open(output_dir / "crop_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nStats saved to {output_dir / 'crop_stats.json'}")


if __name__ == "__main__":
    main()
