import torch
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
from model import HumanSegmentationModel

def load_model(model_path, device='cuda'):
    model = HumanSegmentationModel()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path, img_size=(256, 256)):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    image_resized = image.resize(img_size, Image.BILINEAR)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image_resized)
    return image_tensor, image, original_size

def create_collage(original_image, predicted_mask, img_size=(256, 256)):
    orig_resized = original_image.resize(img_size, Image.BILINEAR)
    
    mask_array = (predicted_mask * 255).astype(np.uint8)
    mask_pil = Image.fromarray(mask_array)
    
    collage_width = img_size[0] * 2
    collage_height = img_size[1]
    collage = Image.new('RGB', (collage_width, collage_height))
    
    collage.paste(orig_resized, (0, 0))
    collage.paste(mask_pil.convert('RGB'), (img_size[0], 0))
    
    from PIL import ImageDraw
    draw = ImageDraw.Draw(collage)
    draw.line([(img_size[0], 0), (img_size[0], img_size[1])], fill='white', width=2)
    
    return collage

def test_model():
    MODEL_PATH = "human_segmentation/best_model.pt"
    IMAGE_FOLDER = "human_segmentation/test"
    OUTPUT_FOLDER = "human_segmentation/out"
    
    device = 'cuda'
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Error: Image folder not found at {IMAGE_FOLDER}")
        return
    
    print(f"Loading model from {MODEL_PATH}")
    model = load_model(MODEL_PATH, device)
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.jfif')
    image_files = [
        f for f in os.listdir(IMAGE_FOLDER) 
        if f.lower().endswith(image_extensions)
    ]
    
    if len(image_files) == 0:
        print(f"Error: No images found in {IMAGE_FOLDER}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        try:
            img_path = os.path.join(IMAGE_FOLDER, img_file)
            print(f"Processing: {img_file}")
            
            image_tensor, original_image, original_size = preprocess_image(img_path)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(image_tensor)
                prediction_binary = (prediction > 0.5).float()
            
            pred_mask = prediction_binary[0, 0].cpu().numpy()
            
            collage = create_collage(original_image, pred_mask)
            
            base_name = os.path.splitext(img_file)[0]
            output_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_result.png")
            collage.save(output_path)
            print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Error processing {img_file}: {str(e)}")
            continue

if __name__ == "__main__":
    test_model()