
import argparse
from pathlib import Path
from PIL import Image
import torch
from src.inference.predictor import load_predictor

def predict_single_image(image_path, model_path, config_path):
    # Load predictor
    print(f"Loading model from {model_path}...")
    predictor = load_predictor(model_path, config_path)
    
    # Load image
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert('RGB')
    
    # Predict
    result = predictor.predict(image)
    
    print("\n" + "="*30)
    print(f"Prediction Result")
    print("="*30)
    print(f"Class: {result['class_name']} (ID: {result['class_id']})")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Confident?: {'Yes' if result['is_confident'] else 'No'}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict traffic sign from image")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--model", default="models_sri_lanka/best_model.pth", help="Path to best_model.pth")
    parser.add_argument("--config", default="config_sri_lanka.yaml", help="Path to config file")
    
    args = parser.parse_args()
    predict_single_image(args.image, args.model, args.config)
