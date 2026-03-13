import torch
import timm
import torchvision.transforms as transforms
from PIL import Image
import io
from ultralytics import YOLO  # Add this for animals

class AgriVisionModel:
    def __init__(self):
        # --- 1. Plant Model (Classification) ---
        self.plant_classes = ["Early Blight", "Late Blight", "Healthy", "Leaf Mold"]
        try:
            self.plant_model = timm.create_model('mobilenetv2_100', pretrained=True, num_classes=4)
            self.plant_model.eval()
            print("Vision Layer: Plant model loaded.")
        except Exception as e:
            print(f"Plant Load failed: {e}")

        # --- 2. Animal Model (Detection) ---
        try:
            # Using 'yolov8n.pt' - it will auto-download or load from your /models folder
            self.animal_model = YOLO('yolov8n.pt') 
            print("Vision Layer: Animal model (YOLO) loaded.")
        except Exception as e:
            print(f"Animal Load failed: {e}")

    def predict_plant(self, image_bytes):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.plant_model(input_tensor)
            probs = torch.nn.functional.softmax(output[0], dim=0)
            conf, idx = torch.max(probs, 0)
        return self.plant_classes[idx.item()], conf.item()

    def predict_livestock(self, image_bytes):
        # Convert bytes to PIL for YOLO
        image = Image.open(io.BytesIO(image_bytes))
        results = self.animal_model(image)[0]
        
        detections = []
        for box in results.boxes:
            # YOLOv8 default classes: 19 is 'cow', 18 is 'sheep', 17 is 'horse'
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if cls in [17, 18, 19]:  # Filter for farm animals
                detections.append({
                    "label": results.names[cls],
                    "confidence": conf,
                    "box": box.xyxy[0].tolist()
                })
        return detections