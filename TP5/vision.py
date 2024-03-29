import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm


def load_model_and_transform():
    """
    Loads a pre-trained ResNet model and defines a transformation to apply on images.

    Returns:
        model (torch.nn.Module): The pre-trained ResNet model.
        transform (torchvision.transforms.Compose): The transformation to apply on images.
        device (torch.device): The device (CPU or GPU) on which the model will be loaded.
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained ResNet
    model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
    model.eval()  # Set the model to evaluation mode

    # Transformation to apply on images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform, device


def extract_embeddings(model, transform, images, device):
    """
    Extracts embeddings from a given model for a list of images.

    Args:
        model (torch.nn.Module): The model used for extracting embeddings.
        transform (torchvision.transforms.Transform): The transformation applied to the images.
        images (List[Union[np.ndarray, PIL.Image.Image]]): The list of images to process.
        device (torch.device): The device to run the model on.

    Returns:
        torch.Tensor: The extracted embeddings as a tensor.
    """
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


# Load the YOLOv5 model
def load_yolo():
    """
    Loads the YOLOv5 model from the ultralytics/yolov5 repository.

    Returns:
        torch.nn.Module: The loaded YOLOv5 model.
    """
    return torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)


def detect_pedestrians(model, image):
    """
    Detects pedestrians in an image using a given model.

    Args:
        model: The object detection model.
        image: The input image.

    Returns:
        A list of detected pedestrians, where each pedestrian is represented as a tuple
        containing the bounding box coordinates (x1, y1, x2, y2) and the confidence score.
    """
    results = model(image)
    pedestrians = results.pred[0]

    # Filter for pedestrians (class 0 in COCO dataset)
    pedestrians = [det for det in pedestrians if int(det[-1]) == 0]
    return pedestrians


def process_frames(model, frame_list):
    """
    Process a list of frames using a given model.

    Args:
        model: The model used for detecting pedestrians.
        frame_list: A list of frames to process.

    Returns:
        None
    """
    with open("../Data/det_yolo.txt", "w") as file:
        # file.write("frame,id,bb_left,bb_top,bb_width,bb_height,conf,x,y,z\n")

        for frame_num, frame in tqdm(enumerate(frame_list)):
            detections = detect_pedestrians(model, frame)

            for det in detections:
                x1, y1, x2, y2, conf, _ = det[:6]
                bb_width, bb_height = x2 - x1, y2 - y1
                x1, y1, bb_width, bb_height = (
                    int(x1),
                    int(y1),
                    int(bb_width),
                    int(bb_height),
                )
                conf = round(float(conf * 100.0), 3)
                file.write(
                    f"{frame_num + 1},-1,{x1},{y1},{bb_width},{bb_height},{conf},-1,-1,-1\n"
                )
