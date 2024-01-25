from scipy.optimize import linear_sum_assignment
import numpy as np

def compute_score(similarity, max_similarity, histogram_score):
    return (similarity / max_similarity) * 0.7 + histogram_score * 0.3

def associate_detections_to_tracks(jaccard_index_frames, similarity_frames, histogram_frames, sigma_iou):
    tracks = {}
    jaccard_values = {}

    # find the maximum value in the keys (which are lists) of similarity_frames
    maximum = np.max([np.max(similarity_frames[key]) for key in similarity_frames.keys()])
    print("maximum dot product : ", maximum)
    for frame in jaccard_index_frames.keys():
        # Convert IoU to cost matrix (1 - IoU)
        score_matrix = compute_score(similarity_frames[frame], maximum, histogram_frames[frame])
        cost_matrix = 1 - score_matrix

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out assignments with a cost higher than the threshold (i.e., IoU lower than sigma_iou)
        matched_indices = [
            (r, c)
            for r, c in zip(row_ind, col_ind)
            if jaccard_index_frames[frame][r, c] >= sigma_iou and score_matrix[r, c] >= 0.5
        ]

        # Extract matches and their IoU values
        matches = [c for r, c in matched_indices]
        iou_values = [score_matrix[r, c] for r, c in matched_indices]

        # Update tracks and jaccard values dictionaries
        tracks[frame] = matches
        jaccard_values[frame] = iou_values

    return tracks, jaccard_values
