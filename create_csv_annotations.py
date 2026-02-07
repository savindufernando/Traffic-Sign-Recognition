import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def create_csvs(data_dir: Path):
    """
    Create Train.csv and Test.csv from YOLO labels and split files.
    Combining existing 'train' and 'val' splits into 'Train.csv' (auto-split by loader),
    and 'test' split into 'Test.csv'.
    """
    splits_dir = data_dir / "splits"
    labels_dir = data_dir / "labels_yolo"
    images_dir = data_dir / "images"
    
    # Verify directories
    if not splits_dir.exists() or not labels_dir.exists():
        print(f"Error: Missing splits or labels in {data_dir}")
        return

    # Helper to parse a split list and return rows
    def process_split(split_names: list, split_tag: str):
        rows = []
        for filename in split_names:
            filename = filename.strip()
            # filename typically 'images/syn_xxxxx.jpg' or just 'syn_xxxxx.jpg'
            # We need the base stem to find label
            file_path = Path(filename)
            stem = file_path.stem
            
            # Label file
            label_file = labels_dir / f"{stem}.txt"
            
            if not label_file.exists():
                print(f"Warning: Label not found for {filename}")
                continue
                
            # Read first class ID (assuming single label for classification training)
            with open(label_file, 'r') as f:
                content = f.read().strip()
                if not content:
                    continue
                class_id = int(content.split()[0])
            
            # Path relative to data_dir
            # user's train.py loads relative to data_dir
            # If data_dir has 'images/' subdir, and filename works.
            # Filenames in splits might have 'images/' prefix (I added that logic).
            # If so, perfect.
            rows.append({
                "Path": filename,
                "ClassId": class_id
            })
        return rows

    # Read splits
    splits = {}
    for split in ["train", "val", "test"]:
        txt_file = splits_dir / f"{split}.txt"
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                splits[split] = [l.strip() for l in f.readlines() if l.strip()]
        else:
            splits[split] = []
            
    print(f"Found splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
    
    # Combine Train + Val -> Train.csv
    train_rows = process_split(splits["train"], "train") + process_split(splits["val"], "val")
    test_rows = process_split(splits["test"], "test")
    
    # Create DataFrames
    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)
    
    # Add dummy ROI columns (for compatibility if needed, though dataset.py ignores them)
    # Width,Height,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2
    defaults = {"Width": 0, "Height": 0, "Roi.X1": 0, "Roi.Y1": 0, "Roi.X2": 0, "Roi.Y2": 0}
    for col, val in defaults.items():
        train_df[col] = val
        test_df[col] = val
        
    # Save CSVs
    train_csv = data_dir / "Train.csv"
    test_csv = data_dir / "Test.csv"
    
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"Saved {train_csv} ({len(train_df)} images)")
    print(f"Saved {test_csv} ({len(test_df)} images)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True, help="Path to synthetic dataset root")
    args = parser.parse_args()
    
    create_csvs(Path(args.data_dir))
