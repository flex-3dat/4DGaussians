
import os
import sys
from rembg import remove
import torch
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

def remove_background_rembg(input_path, output_path):
    input_image = Image.open(input_path)
    output_image = remove(input_image)
    white_bg = Image.new("RGBA", output_image.size, "WHITE")
    white_bg.paste(output_image, (0, 0), output_image)
    white_bg.convert("RGB").save(output_path, "JPEG")

def remove_background_DeepLabV3(input_path, output_path):
    # Load model
    model = models.segmentation.deeplabv3_resnet101(pretrained=True)
    model.eval()

    # Load and preprocess image
    image = Image.open(input_path).convert("RGB")
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Get segmentation
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0)

    # Create mask
    mask = (output_predictions == 15).cpu().numpy().astype('uint8') * 255  # Assuming '15' is the label for 'person'

    # Apply mask to image
    white_bg = Image.new('RGB', image.size, (255, 255, 255))
    image.putalpha(Image.fromarray(mask))
    white_bg.paste(image, (0, 0), image)
    white_bg = white_bg.convert("RGB")
    white_bg.save(output_path, "JPEG")

def batch_process_images(input_folder):
    output_folder = os.path.join(input_folder, 'processed')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()
    
    for filename in tqdm(image_files, desc="Processing images", unit="file"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        remove_background_DeepLabV3(input_path, output_path)
    
    print("Batch processing completed.")





if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python remove_bg.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]
    
    if not os.path.isdir(input_folder):
        print(f"Error: The specified input folder does not exist or is not a directory: {input_folder}")
        sys.exit(1)

    batch_process_images(input_folder)
