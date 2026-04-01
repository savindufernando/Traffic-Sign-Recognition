"""
Extract frames from dashcam MP4 videos for fine-tuning.

Pulls 1 frame per second from each video and saves as JPEG.
Output goes to a NEW directory — never touches existing data.

Usage:
    python scripts/extract_dashcam_frames.py \
        --input "D:/APIIT/FYP/Dashcam Footages" \
        --output "./dashcam_frames" \
        --fps 1
"""

import cv2
import argparse
import sys
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, target_fps: float = 1.0):
    """Extract frames from a single video at the given FPS."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ✗ Cannot open: {video_path.name}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps if video_fps > 0 else 0

    # How many source frames to skip between captures
    frame_interval = max(1, int(video_fps / target_fps))

    saved = 0
    frame_idx = 0
    video_stem = video_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{video_stem}_f{frame_idx:06d}.jpg"
            cv2.imwrite(str(output_dir / filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()
    print(f"  ✓ {video_path.name}: {saved} frames ({duration:.0f}s @ {video_fps:.0f}fps)")
    return saved


def main():
    parser = argparse.ArgumentParser(description="Extract frames from dashcam videos")
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="D:/APIIT/FYP/Dashcam-Footages",
        help="Directory containing MP4 files"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./dashcam_frames",
        help="Output directory for extracted frames (default: ./dashcam_frames)"
    )
    parser.add_argument(
        "--fps", "-f",
        type=float,
        default=1.0,
        help="Frames per second to extract (default: 1)"
    )
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory (never touches existing data)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob("*.MP4")) + sorted(input_dir.glob("*.mp4"))
    print(f"\nFound {len(videos)} video files in {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Target FPS: {args.fps}\n")

    total_saved = 0
    for video in videos:
        saved = extract_frames(video, output_dir, args.fps)
        total_saved += saved

    print(f"\n{'='*50}")
    print(f"Done! Extracted {total_saved} frames total")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
