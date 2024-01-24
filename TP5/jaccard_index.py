import numpy as np
import cv2
import torch
from vision import load_model_and_transform, extract_embeddings
from tqdm import tqdm
# import matplotlib.pyplot as plt

def extract_and_resize(image, bbox, target_size=(224, 224)):
    x, y, w, h = bbox
    x, y, w, h = int(x), int(y), int(w), int(h)
    cropped_image = image[y:y+h, x:x+w]
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
    return resized_image

def process_bboxes(image, bboxes):
    processed_images = []
    for bbox in bboxes:
        cropped_resized_image = extract_and_resize(image, bbox)
        processed_images.append(cropped_resized_image)
    return processed_images

def compute_dot_products(single_embedding, embeddings):
    dot_products = torch.matmul(single_embedding, embeddings.mT)
    return dot_products.squeeze()

def resnet_embedding_similarity_frames(images, frames):
    model, transform, device = load_model_and_transform()
    frame_indices = list(frames.keys())
    similarity_matrices = {}

    # Store embeddings of the previous frame
    previous_embeddings = None
    # processed_images_previous = None

    for frame_index in tqdm(frame_indices):
        bboxes = frames[frame_index]
        num_bboxes_current = len(bboxes)

        if frame_index == frame_indices[0] or num_bboxes_current == 0 or frame_index == frame_indices[-1]:
            # Skip the first frame or if no bounding boxes
            previous_embeddings = None
            # processed_images_previous = None
            continue

        processed_images_current = process_bboxes(images[frame_index], bboxes)
        current_embeddings = extract_embeddings(model, transform, processed_images_current, device)

        if previous_embeddings is not None:
            num_bboxes_previous = previous_embeddings.shape[0]
            similarity_matrix = np.zeros((num_bboxes_current, num_bboxes_previous))

            for i in range(num_bboxes_current):
                # subplot single image with image and their dot product
                similarity_matrix[i] = compute_dot_products(current_embeddings[i], previous_embeddings).cpu().numpy()
                
                # plot the current image and the previous images it is comparing
                # to and their dot products
                # if num_bboxes_previous <= 1:
                #     continue
                # fig, axs = plt.subplots(1, num_bboxes_previous + 1)
                # axs[0].imshow(processed_images_current[i])
                # axs[0].set_title("Current")
                # axs[0].axis('off')
                # for j in range(num_bboxes_previous):
                #     axs[j + 1].imshow(processed_images_previous[j])
                #     axs[j + 1].axis('off')
                #     axs[j + 1].set_title(similarity_matrix[i][j])
                # plt.show()

            similarity_matrices[frame_index] = similarity_matrix

        # Update previous_embeddings for the next iteration
        previous_embeddings = current_embeddings
        # processed_images_previous = processed_images_current

    return similarity_matrices