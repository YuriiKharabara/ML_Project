import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.ndimage import label
import pytesseract
import torch
import torch.nn as nn
from src.UNET import UNet as SegmentationModel
from src.UIElementClassifier import UIElementClassifier
import torch
from PIL import Image, ImageDraw
import numpy as np
from torchvision import transforms
from src.UNET import UNet as SegmentationModel
from src.UIElementClassifier import UIElementClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_model(model_class, checkpoint_path, num_classes, device):
    if model_class == SegmentationModel:
        model = model_class(n_channels=3, n_classes=1, bilinear=True)
    else:
        model = model_class(num_classes=num_classes)

    model = model.to(device)

    if model_class == SegmentationModel and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
def get_average_color(image):
    np_image = np.array(image)
    # Calculate the average color along the spatial dimensions
    # and keep the color channels separate
    average_color = np_image.mean(axis=(0, 1)).astype(int)
    # Convert the average color to hex format
    hex_color = '#{:02x}{:02x}{:02x}'.format(*average_color)
    return hex_color


def extract_text(image, coords):
    x1, y1, x2, y2 = coords
    x1 = max(0, min(x1, image.width - 1))
    y1 = max(0, min(y1, image.height - 1))
    x2 = max(x1 + 1, min(x2, image.width))
    y2 = max(y1 + 1, min(y2, image.height))
    cropped_image = image.crop((x1, y1, x2, y2))

    # pytesseract to extract text
    text = pytesseract.image_to_string(cropped_image, lang='eng')
    return text

def extract_ui_elements(image_path, segmentation_model, classification_model, device):
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Segmentation
    with torch.no_grad():
        masks = segmentation_model(input_tensor)
        masks = torch.sigmoid(masks)
        masks = (masks > 0.8).float()  # Threshold, can be changed if needed

    # Prepare for classification
    transform_resize = transforms.Resize((64, 64))
    label_mapping = {0: 'AXStaticText', 1: 'AXButton', 2: 'AXImage'}
    ui_elements = []

    # Find connected components for precise bounding boxes
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(masks.cpu().squeeze(0).squeeze(0).numpy(), structure=structure)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    for i in range(1, num_features + 1):
        ys, xs = np.where(labeled == i)
        if ys.size > 0 and xs.size > 0:
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()

            # Crop and classify
            cropped_img = image.crop((x1, y1, x2, y2))
            cropped_tensor = transform_resize(cropped_img)
            cropped_tensor = transform(cropped_tensor).unsqueeze(0).to(device)

            with torch.no_grad():
                output = classification_model(cropped_tensor)
                _, predicted = torch.max(output, 1)
                element_type = label_mapping[predicted.item()]

            color = get_average_color(cropped_img)

            # Scale coordinates back to original size
            scale_x, scale_y = original_size[0] / 640, original_size[1] / 640
            scaled_coords = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
            try:
                text = extract_text(image, scaled_coords)
            except:
                text = ""

            # detected element information
            ui_elements.append({
                "type": element_type,
                "coords": scaled_coords,
                "color": color,
                "text": text
            })

    return ui_elements