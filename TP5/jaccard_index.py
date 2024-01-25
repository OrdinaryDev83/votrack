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

def compute_iou(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # compute the intersection rectangle
    xA = max(x1, x2)
    yA = max(y1, y2)
    xB = min(x1 + w1, x2 + w2)
    yB = min(y1 + h1, y2 + h2)
 
    # compute the area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    bbox1_area = (w1 + 1) * (h1 + 1)
    bbox2_area = (w2 + 1) * (h2 + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
 
    # return the intersection over union value
    return iou

def compute_dot_products(single_embedding, embeddings):
    dot_products = torch.matmul(single_embedding, embeddings.mT)
    return dot_products.squeeze()

def compute_histograms(a, b, num_bins=10):
    # Compute the histograms of the arrays
    a_hist, _ = np.histogram(a, bins=num_bins, range=(0, 1), density=True)
    b_hist, _ = np.histogram(b, bins=num_bins, range=(0, 1), density=True)

    # Reshape histograms to 2D arrays (1 row, num_bins columns)
    a_hist = a_hist.reshape((1, -1)).astype(np.float32)
    b_hist = b_hist.reshape((1, -1)).astype(np.float32)

    return a_hist, b_hist

def compare_histograms(a_hist, b_hist, method=cv2.HISTCMP_CORREL):
    # compare the histograms
    score = cv2.compareHist(a_hist, b_hist, method)
    return score

def resnet_embedding_similarity_frames(images, frames):
    model, transform, device = load_model_and_transform()
    frame_indices = list(frames.keys())
    similarity_matrices = {}
    jaccard_indices = {}
    histograms_indices = {}

    # Store embeddings of the previous frame
    previous_embeddings = None
    processed_images_previous = None

    for frame_index in tqdm(frame_indices):
        previous_bboxes = frames[frame_indices[frame_indices.index(frame_index) - 1]]
        bboxes = frames[frame_index]
        num_bboxes_current = len(bboxes)

        if frame_index == frame_indices[0] or num_bboxes_current == 0 or frame_index == frame_indices[-1]:
            # Skip the first frame or if no bounding boxes
            previous_embeddings = None
            processed_images_previous = None
            continue

        processed_images_current = process_bboxes(images[frame_index], bboxes)
        current_embeddings = extract_embeddings(model, transform, processed_images_current, device)

        if previous_embeddings is not None:
            num_bboxes_previous = previous_embeddings.shape[0]
            similarity_matrix = np.zeros((num_bboxes_current, num_bboxes_previous))
            jaccard_index = np.zeros((num_bboxes_current, num_bboxes_previous))
            histograms_index = np.zeros((num_bboxes_current, num_bboxes_previous))

            for i in range(num_bboxes_current):
                # subplot single image with image and their dot product
                similarity_matrix[i] = compute_dot_products(current_embeddings[i], previous_embeddings).cpu().numpy()
                for j in range(num_bboxes_previous):
                    # compute iou
                    jaccard_index[i][j] = compute_iou(bboxes[i], previous_bboxes[j])
                    # compute histograms
                    a_hist, b_hist = compute_histograms(processed_images_current[i], processed_images_previous[j])
                    histograms_index[i][j] = compare_histograms(a_hist, b_hist)
                
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
            jaccard_indices[frame_index] = jaccard_index
            histograms_indices[frame_index] = histograms_index

        # Update previous_embeddings for the next iteration
        previous_embeddings = current_embeddings
        processed_images_previous = processed_images_current

    return jaccard_indices, similarity_matrices, histograms_indices