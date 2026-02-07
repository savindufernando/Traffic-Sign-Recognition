#!/usr/bin/env python
"""
Generate Synthetic Dataset for Sri Lankan Traffic Signs.

This is the main entry point for the synthetic data generation pipeline.
It orchestrates the entire process from template loading to annotation export.

Usage:
    python generate_dataset.py --templates ./sri_lankan_traffc_signs/raw \
                               --backgrounds ./sri_lankan_traffc_signs/backgrounds \
                               --output ./sri_lankan_traffc_signs/synthetic \
                               --n-per-template 100 \
                               --seed 42 \
                               --formats yolo coco voc
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.data.synthetic_data_generator import SyntheticDataGenerator
from src.data.annotation_exporter import (
    AnnotationExporter,
    BoundingBox,
    create_class_list_from_templates,
    create_train_val_test_split
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic traffic sign dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python generate_dataset.py \\
        --templates ./sri_lankan_traffc_signs/raw \\
        --backgrounds ./sri_lankan_traffc_signs/backgrounds \\
        --output ./sri_lankan_traffc_signs/synthetic \\
        --n-per-template 100 \\
        --seed 42

Note: 
    Before running, you must generate background images using the prompts
    in sri_lankan_traffc_signs/background_prompts.md
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--templates", "-t",
        type=str,
        default="./sri_lankan_traffc_signs/raw",
        help="Path to template images directory (default: ./sri_lankan_traffc_signs/raw)"
    )
    parser.add_argument(
        "--backgrounds", "-b",
        type=str,
        default="./sri_lankan_traffc_signs/raw/backgrounds",
        help="Path to background images directory (default: ./sri_lankan_traffc_signs/raw/backgrounds)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="./sri_lankan_traffc_signs/synthetic",
        help="Output directory for generated data (default: ./sri_lankan_traffc_signs/synthetic)"
    )
    
    # Generation options
    parser.add_argument(
        "--n-per-template", "-n",
        type=int,
        default=100,
        help="Number of variations per template image (default: 100)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["yolo", "coco", "voc"],
        default=["yolo", "coco", "voc"],
        help="Annotation formats to export (default: all)"
    )
    
    # Optional features
    parser.add_argument(
        "--no-hard-negatives",
        action="store_true",
        help="Disable hard negative sample generation"
    )
    parser.add_argument(
        "--bg-category",
        type=str,
        default=None,
        help="Use only specific background category (e.g., 'urban_day', 'rain', 'night')"
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=0.1,
        help="Ratio of hard negatives to positive samples (default: 0.1)"
    )
    parser.add_argument(
        "--add-annotation-noise",
        action="store_true",
        help="Add annotation noise for model robustness"
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Skip creating train/val/test splits"
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=None,
        help="Preview mode: generate only N images for quality check (e.g., --preview 10)"
    )
    parser.add_argument(
        "--no-poles",
        action="store_true",
        help="Disable sign pole rendering"
    )
    parser.add_argument(
        "--exhaustive", "-e",
        action="store_true",
        help="Generate every sign with every background (guarantees coverage)"
    )
    
    # Split ratios
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation set ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio (default: 0.15)"
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Convert paths
    template_dir = Path(args.templates)
    background_dir = Path(args.backgrounds)
    output_dir = Path(args.output)
    
    # Filter to specific background category if specified
    if args.bg_category:
        background_dir = background_dir / args.bg_category
        print(f"Using only '{args.bg_category}' backgrounds")
    
    # Validate paths
    if not template_dir.exists():
        logger.error(f"Template directory not found: {template_dir}")
        logger.info("Please ensure your template images are in the correct location.")
        sys.exit(1)
    
    if not background_dir.exists():
        logger.error(f"Background directory not found: {background_dir}")
        logger.info("\n" + "="*60)
        logger.info("IMPORTANT: You need to generate background images first!")
        logger.info("="*60)
        logger.info("\n1. Open: sri_lankan_traffc_signs/background_prompts.md")
        logger.info("2. Use the prompts with Grok/DALL-E/Midjourney")
        logger.info("3. Save generated images in the backgrounds directory:")
        logger.info(f"   {background_dir}")
        logger.info("   - organized by folder (urban_day, rain, night, etc.)")
        sys.exit(1)
    
    # Check for actual background images
    bg_count = len(list(background_dir.rglob("*.jpg"))) + \
               len(list(background_dir.rglob("*.png"))) + \
               len(list(background_dir.rglob("*.jpeg")))
    
    if bg_count == 0:
        logger.error(f"No background images found in: {background_dir}")
        logger.info("\nPlease generate backgrounds using the prompts in:")
        logger.info("  sri_lankan_traffc_signs/background_prompts.md")
        sys.exit(1)
    
    logger.info(f"Found {bg_count} background images")
    
    print("\n" + "="*60)
    print("SRI LANKAN TRAFFIC SIGN - SYNTHETIC DATASET GENERATOR")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Templates:        {template_dir}")
    print(f"  Backgrounds:      {background_dir} ({bg_count} images)")
    print(f"  Output:           {output_dir}")
    print(f"  Variations/template: {args.n_per_template}")
    print(f"  Random seed:      {args.seed}")
    print(f"  Export formats:   {', '.join(args.formats)}")
    print(f"  Hard negatives:   {'Disabled' if args.no_hard_negatives else f'Enabled ({args.hard_negative_ratio*100:.0f}%)'}")
    print(f"  Annotation noise: {'Enabled' if args.add_annotation_noise else 'Disabled'}")
    print(f"  Sign poles:       {'Disabled' if args.no_poles else 'Enabled'}")
    if args.preview:
        print(f"  *** PREVIEW MODE: Generating only {args.preview} images ***")
    print("\n" + "-"*60 + "\n")
    
    # Step 1: Generate synthetic images
    print("[1/4] Generating synthetic images...")
    generator = SyntheticDataGenerator(
        template_dir=template_dir,
        background_dir=background_dir,
        output_dir=output_dir,
        seed=args.seed
    )
    
    # Determine n_per_template (respect preview mode)
    n_per_template = args.preview if args.preview else args.n_per_template
    
    stats = generator.generate_dataset(
        n_per_template=n_per_template,
        include_hard_negatives=not args.no_hard_negatives and not args.preview,
        hard_negative_ratio=args.hard_negative_ratio,
        exhaustive_mode=args.exhaustive
    )
    
    print(f"\n✓ Generated {stats['total_generated']} images")
    print(f"  Rejected: {stats['rejected']}")
    print(f"  Hard negatives: {stats['hard_negatives']}")
    
    # Step 2: Export annotations
    print("\n[2/4] Exporting annotations...")
    
    # Get class names from templates
    class_names = create_class_list_from_templates(template_dir / "online_collected")
    print(f"  Found {len(class_names)} sign classes")
    
    # Create exporter
    exporter = AnnotationExporter(
        output_dir=output_dir,
        class_names=class_names,
        add_noise=args.add_annotation_noise,
        noise_seed=args.seed
    )
    
    # Load metadata
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Export annotations for each image
    for entry in metadata:
        if entry.get("is_hard_negative", False):
            continue
        
        bbox = entry["bbox"]
        class_name = Path(entry["source_template"]).stem.lower().replace(" ", "_").replace("-", "_")
        
        if class_name not in exporter.class_to_id:
            # Try to find partial match
            matching = [c for c in class_names if class_name.startswith(c) or c.startswith(class_name)]
            if matching:
                class_name = matching[0]
            else:
                continue
        
        bb = BoundingBox(
            x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
            class_id=exporter.class_to_id.get(class_name, 0),
            class_name=class_name
        )
        
        # Export based on requested formats
        if "yolo" in args.formats:
            exporter.export_yolo(entry["output_file"], [bb], 1920, 1080)
        if "voc" in args.formats:
            exporter.export_voc(entry["output_file"], [bb], 1920, 1080)
        if "coco" in args.formats:
            exporter.export_coco_entry(entry["output_file"], [bb], 1920, 1080)
    
    # Finalize COCO
    if "coco" in args.formats:
        exporter.finalize_coco()
    
    # Save class names
    class_files = exporter.save_class_names()
    
    print("✓ Annotations exported:")
    for fmt in args.formats:
        if fmt == "yolo":
            print(f"    YOLO: {exporter.yolo_dir}")
        elif fmt == "coco":
            print(f"    COCO: {exporter.coco_dir}")
        elif fmt == "voc":
            print(f"    VOC:  {exporter.voc_dir}")
    
    # Step 3: Create train/val/test splits
    if not args.no_split:
        print("\n[3/4] Creating train/val/test splits...")
        splits = create_train_val_test_split(
            image_dir=output_dir / "images",
            output_dir=output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print("✓ Splits created")
    else:
        print("\n[3/4] Skipping splits (--no-split flag)")
    
    # Step 4: Generate summary report
    print("\n[4/4] Generating summary report...")
    
    report = {
        "dataset_name": "Sri Lankan Traffic Sign Dataset - Synthetic",
        "version": "1.0",
        "generated_at": datetime.now().isoformat(),
        "configuration": {
            "n_per_template": args.n_per_template,
            "seed": args.seed,
            "formats": args.formats,
            "hard_negatives": not args.no_hard_negatives,
            "annotation_noise": args.add_annotation_noise
        },
        "statistics": {
            "total_images": stats["total_generated"],
            "hard_negatives": stats["hard_negatives"],
            "rejected": stats["rejected"],
            "num_classes": len(class_names),
            "per_category": stats["per_category"]
        },
        "paths": {
            "images": str(output_dir / "images"),
            "yolo_labels": str(exporter.yolo_dir) if "yolo" in args.formats else None,
            "coco_labels": str(exporter.coco_dir) if "coco" in args.formats else None,
            "voc_labels": str(exporter.voc_dir) if "voc" in args.formats else None,
            "splits": str(output_dir / "splits") if not args.no_split else None
        }
    }
    
    report_path = output_dir / "dataset_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print("✓ Report saved to", report_path)
    
    # Final summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE!")
    print("="*60)
    print(f"\n📁 Output directory: {output_dir}")
    print(f"\n📊 Statistics:")
    print(f"   • Total images:    {stats['total_generated']}")
    print(f"   • Hard negatives:  {stats['hard_negatives']}")
    print(f"   • Classes:         {len(class_names)}")
    print(f"\n📝 Next steps:")
    print(f"   1. Review generated images in: {output_dir / 'images'}")
    print(f"   2. Check annotations in your preferred format")
    print(f"   3. Train your model using:")
    print(f"      python train.py --data_path {output_dir}")
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()
