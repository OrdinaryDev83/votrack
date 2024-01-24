import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np

def load_model_and_transform():
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()  # Set the model to evaluation mode

    # Transformation to apply on images
    transform = transforms.Compose([
        transforms.Resize((224, 224), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, device

def extract_embeddings(model, transform, images, device):
    # Process images and extract embeddings
    embeddings = []

    with torch.no_grad():
        for img in images:
            img_pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
            img_t = transform(img_pil).to(device)
            img_t = img_t.unsqueeze(0)  # Add batch dimension
            embedding = model(img_t)
            embeddings.append(embedding)

    embeddings = torch.cat(embeddings, dim=0)
    return embeddings