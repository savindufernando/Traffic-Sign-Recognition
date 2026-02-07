"""Quick debug script to test model predictions directly."""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PIL import Image
from src.inference.predictor import load_predictor

def test_prediction():
    # Load model
    model_path = project_root / "models_sri_lanka" / "best_model.pth"
    config_path = project_root / "config_sri_lanka.yaml"
    
    print("Loading model...")
    predictor = load_predictor(str(model_path), str(config_path))
    print(f"Model loaded! Number of classes: {len(predictor.class_names)}")
    print(f"First 10 class names: {predictor.class_names[:10]}")
    print()
    
    # Test with synthetic images known to be in Train.csv
    test_images = [
        "sri_lankan_traffc_signs/synthetic/images/syn_000095.jpg", # Class 84 (accident)
        "sri_lankan_traffc_signs/synthetic/images/syn_000019.jpg", # Class 83 (yellow_traffic_light)
        "sri_lankan_traffc_signs/synthetic/images/syn_000023.jpg", # Class 83 (yellow_traffic_light)
    ]
    
    # Get expected classes from Train.csv
    import csv
    expected_classes = {}
    csv_path = project_root / "sri_lankan_traffc_signs/synthetic/Train.csv"
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path = row['Path']
            class_id = int(row['ClassId'])
            expected_classes[path] = class_id
    
    for img_path in test_images:
        full_path = project_root / img_path
        if full_path.exists():
            print(f"Testing: {img_path}")
            
            # Get expected class
            csv_key = "images/" + Path(img_path).name
            expected_id = expected_classes.get(csv_key, "Unknown")
            expected_name = predictor.class_names[expected_id] if isinstance(expected_id, int) else "Unknown"
            print(f"  Expected: Class {expected_id} = '{expected_name}'")
            
            # Get prediction
            image = Image.open(full_path).convert('RGB')
            result = predictor.predict(image)
            
            print(f"  Predicted: Class {result['class_id']} = '{result['class_name']}'")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Match: {'✅' if result['class_id'] == expected_id else '❌'}")
            print()

if __name__ == "__main__":
    test_prediction()
