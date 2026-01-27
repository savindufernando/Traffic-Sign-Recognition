"""
Multi-Format Annotation Exporter for Synthetic Traffic Sign Dataset.

Exports annotations in:
- YOLO format (.txt per image)
- COCO format (single annotations.json)
- Pascal VOC format (.xml per image)

Includes annotation noise simulation for model robustness.
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import random
from datetime import datetime


@dataclass
class BoundingBox:
    """Bounding box representation."""
    x: int  # Left
    y: int  # Top
    w: int  # Width
    h: int  # Height
    class_id: int
    class_name: str
    
    def to_yolo(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format: class_id x_center y_center width height (normalized)."""
        x_center = (self.x + self.w / 2) / img_width
        y_center = (self.y + self.h / 2) / img_height
        w_norm = self.w / img_width
        h_norm = self.h / img_height
        return f"{self.class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"
    
    def to_coco(self) -> List[float]:
        """Convert to COCO format: [x, y, width, height]."""
        return [float(self.x), float(self.y), float(self.w), float(self.h)]
    
    def to_voc(self) -> Dict[str, int]:
        """Convert to Pascal VOC format: xmin, ymin, xmax, ymax."""
        return {
            "xmin": self.x,
            "ymin": self.y,
            "xmax": self.x + self.w,
            "ymax": self.y + self.h
        }


class AnnotationNoiseSimulator:
    """
    Simulates annotation noise (human labeling errors) for model robustness.
    Adds small random jitter to bounding boxes.
    """
    
    def __init__(
        self, 
        jitter_ratio: float = 0.02,
        noise_probability: float = 0.3,
        seed: Optional[int] = None
    ):
        """
        Args:
            jitter_ratio: Maximum jitter as fraction of bbox size
            noise_probability: Probability of applying noise to each bbox
            seed: Random seed for reproducibility
        """
        self.jitter_ratio = jitter_ratio
        self.noise_probability = noise_probability
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def add_noise(self, bbox: BoundingBox, img_width: int, img_height: int) -> BoundingBox:
        """Add annotation noise to bounding box."""
        if random.random() > self.noise_probability:
            return bbox
            
        # Calculate jitter amounts
        x_jitter = int(bbox.w * self.jitter_ratio * np.random.uniform(-1, 1))
        y_jitter = int(bbox.h * self.jitter_ratio * np.random.uniform(-1, 1))
        w_jitter = int(bbox.w * self.jitter_ratio * np.random.uniform(-1, 1))
        h_jitter = int(bbox.h * self.jitter_ratio * np.random.uniform(-1, 1))
        
        # Apply jitter with bounds checking
        new_x = max(0, bbox.x + x_jitter)
        new_y = max(0, bbox.y + y_jitter)
        new_w = max(10, bbox.w + w_jitter)
        new_h = max(10, bbox.h + h_jitter)
        
        # Ensure bbox stays within image
        new_w = min(new_w, img_width - new_x)
        new_h = min(new_h, img_height - new_y)
        
        return BoundingBox(
            x=new_x, y=new_y, w=new_w, h=new_h,
            class_id=bbox.class_id, class_name=bbox.class_name
        )


class AnnotationExporter:
    """
    Exports annotations in multiple formats.
    """
    
    def __init__(
        self,
        output_dir: Path,
        class_names: List[str],
        image_width: int = 1920,
        image_height: int = 1080,
        add_noise: bool = True,
        noise_seed: Optional[int] = None
    ):
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.class_to_id = {name: idx for idx, name in enumerate(class_names)}
        self.image_width = image_width
        self.image_height = image_height
        
        # Create output directories
        self.yolo_dir = self.output_dir / "labels_yolo"
        self.coco_dir = self.output_dir / "labels_coco"
        self.voc_dir = self.output_dir / "labels_voc"
        
        for d in [self.yolo_dir, self.coco_dir, self.voc_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Noise simulator
        self.noise_simulator = AnnotationNoiseSimulator(seed=noise_seed) if add_noise else None
        
        # COCO accumulator
        self.coco_annotations = {
            "info": {
                "description": "Sri Lankan Traffic Sign Dataset - Synthetic",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "Synthetic Data Generator",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [{"id": 1, "name": "Research Use", "url": ""}],
            "categories": [
                {"id": idx, "name": name, "supercategory": "traffic_sign"}
                for idx, name in enumerate(class_names)
            ],
            "images": [],
            "annotations": []
        }
        self.coco_image_id = 0
        self.coco_annotation_id = 0
    
    def _apply_noise_if_enabled(
        self, 
        bbox: BoundingBox, 
        img_w: int, 
        img_h: int
    ) -> BoundingBox:
        """Apply noise if enabled."""
        if self.noise_simulator:
            return self.noise_simulator.add_noise(bbox, img_w, img_h)
        return bbox
    
    def export_yolo(
        self,
        image_filename: str,
        bboxes: List[BoundingBox],
        img_width: int,
        img_height: int
    ) -> Path:
        """
        Export annotations in YOLO format.
        
        Args:
            image_filename: Name of the image file
            bboxes: List of bounding boxes
            img_width: Image width
            img_height: Image height
            
        Returns:
            Path to the created annotation file
        """
        label_filename = Path(image_filename).stem + ".txt"
        label_path = self.yolo_dir / label_filename
        
        lines = []
        for bbox in bboxes:
            noisy_bbox = self._apply_noise_if_enabled(bbox, img_width, img_height)
            lines.append(noisy_bbox.to_yolo(img_width, img_height))
        
        with open(label_path, 'w') as f:
            f.write("\n".join(lines))
            
        return label_path
    
    def export_coco_entry(
        self,
        image_filename: str,
        bboxes: List[BoundingBox],
        img_width: int,
        img_height: int
    ):
        """
        Add annotations for an image to COCO accumulator.
        Call finalize_coco() after processing all images.
        """
        # Add image entry
        self.coco_annotations["images"].append({
            "id": self.coco_image_id,
            "file_name": image_filename,
            "width": img_width,
            "height": img_height
        })
        
        # Add annotation entries
        for bbox in bboxes:
            noisy_bbox = self._apply_noise_if_enabled(bbox, img_width, img_height)
            
            self.coco_annotations["annotations"].append({
                "id": self.coco_annotation_id,
                "image_id": self.coco_image_id,
                "category_id": noisy_bbox.class_id,
                "bbox": noisy_bbox.to_coco(),
                "area": noisy_bbox.w * noisy_bbox.h,
                "iscrowd": 0,
                "segmentation": []
            })
            self.coco_annotation_id += 1
        
        self.coco_image_id += 1
    
    def finalize_coco(self) -> Path:
        """
        Write accumulated COCO annotations to file.
        
        Returns:
            Path to the created annotations.json file
        """
        output_path = self.coco_dir / "annotations.json"
        with open(output_path, 'w') as f:
            json.dump(self.coco_annotations, f, indent=2)
        return output_path
    
    def export_voc(
        self,
        image_filename: str,
        bboxes: List[BoundingBox],
        img_width: int,
        img_height: int
    ) -> Path:
        """
        Export annotations in Pascal VOC XML format.
        
        Args:
            image_filename: Name of the image file
            bboxes: List of bounding boxes
            img_width: Image width
            img_height: Image height
            
        Returns:
            Path to the created annotation file
        """
        # Create root element
        annotation = ET.Element("annotation")
        
        # Basic info
        ET.SubElement(annotation, "folder").text = "images"
        ET.SubElement(annotation, "filename").text = image_filename
        
        # Source
        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Sri Lankan Traffic Sign Dataset"
        ET.SubElement(source, "annotation").text = "Synthetic"
        
        # Size
        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img_width)
        ET.SubElement(size, "height").text = str(img_height)
        ET.SubElement(size, "depth").text = "3"
        
        ET.SubElement(annotation, "segmented").text = "0"
        
        # Objects (bounding boxes)
        for bbox in bboxes:
            noisy_bbox = self._apply_noise_if_enabled(bbox, img_width, img_height)
            voc_coords = noisy_bbox.to_voc()
            
            obj = ET.SubElement(annotation, "object")
            ET.SubElement(obj, "name").text = noisy_bbox.class_name
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(voc_coords["xmin"])
            ET.SubElement(bndbox, "ymin").text = str(voc_coords["ymin"])
            ET.SubElement(bndbox, "xmax").text = str(voc_coords["xmax"])
            ET.SubElement(bndbox, "ymax").text = str(voc_coords["ymax"])
        
        # Pretty print XML
        xml_str = minidom.parseString(ET.tostring(annotation)).toprettyxml(indent="  ")
        
        # Save
        label_filename = Path(image_filename).stem + ".xml"
        label_path = self.voc_dir / label_filename
        
        with open(label_path, 'w') as f:
            f.write(xml_str)
            
        return label_path
    
    def export_all_formats(
        self,
        image_filename: str,
        bboxes: List[BoundingBox],
        img_width: int,
        img_height: int
    ) -> Dict[str, Path]:
        """
        Export annotations in all formats simultaneously.
        
        Returns:
            Dictionary mapping format name to output path
        """
        return {
            "yolo": self.export_yolo(image_filename, bboxes, img_width, img_height),
            "voc": self.export_voc(image_filename, bboxes, img_width, img_height),
            "coco": None  # COCO is accumulated and finalized separately
        }
    
    def save_class_names(self) -> Dict[str, Path]:
        """
        Save class names file for YOLO training.
        
        Returns:
            Paths to created files
        """
        # classes.txt for YOLO
        yolo_classes = self.yolo_dir / "classes.txt"
        with open(yolo_classes, 'w') as f:
            f.write("\n".join(self.class_names))
        
        # classes.yaml for YOLOv5/v8
        yaml_content = f"""# Sri Lankan Traffic Sign Dataset
# Auto-generated by synthetic data generator

path: {self.output_dir.absolute()}
train: images/train
val: images/val
test: images/test

names:
"""
        for idx, name in enumerate(self.class_names):
            yaml_content += f"  {idx}: {name}\n"
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)
        
        return {
            "classes_txt": yolo_classes,
            "dataset_yaml": yaml_path
        }


def create_class_list_from_templates(template_dir: Path) -> List[str]:
    """
    Create class list from template directory structure.
    
    Args:
        template_dir: Path to template directory
        
    Returns:
        List of class names
    """
    class_names = []
    
    # Walk through template directory
    for item in sorted(template_dir.rglob("*.png")):
        class_name = item.stem.lower().replace(" ", "_").replace("-", "_")
        if class_name not in class_names:
            class_names.append(class_name)
    
    for item in sorted(template_dir.rglob("*.jpg")):
        class_name = item.stem.lower().replace(" ", "_").replace("-", "_")
        if class_name not in class_names:
            class_names.append(class_name)
    
    return class_names


# =============================================================================
# TRAIN/VAL/TEST SPLIT UTILITY
# =============================================================================

def create_train_val_test_split(
    image_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Dict[str, List[str]]:
    """
    Create train/val/test split files.
    
    Args:
        image_dir: Directory containing images
        output_dir: Output directory for split files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        
    Returns:
        Dictionary with lists of filenames for each split
    """
    np.random.seed(seed)
    
    # Get all image files
    images = sorted(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
    image_names = [img.name for img in images]
    
    # Shuffle
    np.random.shuffle(image_names)
    
    # Calculate split indices
    n = len(image_names)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    splits = {
        "train": image_names[:train_end],
        "val": image_names[train_end:val_end],
        "test": image_names[val_end:]
    }
    
    # Save split files
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, file_list in splits.items():
        split_path = splits_dir / f"{split_name}.txt"
        with open(split_path, 'w') as f:
            f.write("\n".join(file_list))
    
    # Print statistics
    print(f"\nDataset Split Statistics:")
    print(f"  Train: {len(splits['train'])} images ({len(splits['train'])/n*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} images ({len(splits['val'])/n*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} images ({len(splits['test'])/n*100:.1f}%)")
    
    return splits


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Export annotations in multiple formats")
    parser.add_argument("--metadata", type=str, required=True,
                       help="Path to metadata.json from generator")
    parser.add_argument("--output", type=str, required=True,
                       help="Output directory")
    parser.add_argument("--templates", type=str, required=True,
                       help="Path to templates directory for class names")
    parser.add_argument("--add-noise", action="store_true",
                       help="Add annotation noise for robustness")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Load metadata
    with open(args.metadata, 'r') as f:
        metadata = json.load(f)
    
    # Get class names
    class_names = create_class_list_from_templates(Path(args.templates))
    print(f"Found {len(class_names)} classes")
    
    # Create exporter
    exporter = AnnotationExporter(
        output_dir=Path(args.output),
        class_names=class_names,
        add_noise=args.add_noise,
        noise_seed=args.seed
    )
    
    # Export annotations for each image
    for entry in metadata:
        if entry.get("is_hard_negative", False):
            continue
            
        bbox = entry["bbox"]
        class_name = Path(entry["source_template"]).stem.lower().replace(" ", "_")
        
        if class_name not in exporter.class_to_id:
            print(f"Warning: Unknown class '{class_name}'")
            continue
        
        bb = BoundingBox(
            x=bbox[0], y=bbox[1], w=bbox[2], h=bbox[3],
            class_id=exporter.class_to_id[class_name],
            class_name=class_name
        )
        
        exporter.export_all_formats(
            entry["output_file"],
            [bb],
            1920, 1080  # Default size
        )
        exporter.export_coco_entry(entry["output_file"], [bb], 1920, 1080)
    
    # Finalize COCO
    exporter.finalize_coco()
    exporter.save_class_names()
    
    print(f"\nExported annotations to {args.output}")
    print(f"  - YOLO: {exporter.yolo_dir}")
    print(f"  - COCO: {exporter.coco_dir}")
    print(f"  - VOC:  {exporter.voc_dir}")
