"""
Automated YOLOv8 Training and Export Script
This script trains a YOLOv8 Nano model on the synthetic Sri Lankan traffic sign dataset,
and then automatically exports the weights to TensorFlow.js (tfjs) format for the React frontend.
"""

import sys
import subprocess
from pathlib import Path

def check_ultralytics():
    try:
        import ultralytics
        print(f"Ultralytics is installed: {ultralytics.__version__}")
    except ImportError:
        print("Ultralytics is not installed. Installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        print("Ultralytics installed successfully.")

def train_and_export():
    from ultralytics import YOLO

    dataset_yaml = Path(r"D:\APIIT\FYP\Traffic-Sign-Recognition\sri_lankan_traffc_signs\synthetic\dataset.yaml")
    
    if not dataset_yaml.exists():
        print(f"Error: Dataset not found at {dataset_yaml}")
        return

    # Load a pre-trained YOLOv8 Nano model
    print("Loading YOLOv8n pre-trained weights...")
    model = YOLO("yolov8n.pt")  

    # Train the model on the synthetic dataset
    # You can change epochs depending on your compute power and time constraints
    epochs = 50 
    
    print(f"Starting training on {dataset_yaml} for {epochs} epochs...")
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=320,  # Reduced from 640 to 320. This makes it 4x faster!
        workers=2,  # Added multi-threading to speed up image loading
        project="yolo_training",
        name="sri_lanka_signs",
        device="cpu"  # Change to "0" if you have an Nvidia GPU installed with CUDA!
    )

    print("\nTraining completed successfully!")
    
    # Get the best trained weights
    best_weights = model.trainer.best
    print(f"Best weights saved at: {best_weights}")

    # Load the custom trained model
    custom_model = YOLO(best_weights)

    # Export to ONNX format
    print("\nExporting model to ONNX format...")
    export_path = custom_model.export(format="onnx")
    
    print("\n" + "="*60)
    print("EXPORT COMPLETE!")
    print(f"Your React-ready file is located at: {export_path}")
    print("Next Steps:")
    print("1. Copy the 'best.onnx' file from that folder.")
    print("2. Paste it into D:\\APIIT\\FYP\\FusionLayer\\dashboard\\public\\model\\ (Replace the old best.onnx)")
    print("3. Restart your React app!")
    print("="*60)

if __name__ == "__main__":
    check_ultralytics()
    train_and_export()
