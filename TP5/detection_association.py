from scipy.optimize import linear_sum_assignment
import numpy as np


def associate_detections_to_tracks(jaccard_index_frames, sigma_iou):
    tracks = {}
    jaccard_values = {}

    # find the maximum value in the keys (which are lists) of jaccard_index_frames
    maximum = np.max([np.max(jaccard_index_frames[key]) for key in jaccard_index_frames.keys()])
    print("maximum dot product : ", maximum)
    for frame in jaccard_index_frames.keys():
        # Convert IoU to cost matrix (1 - IoU)
        cost_matrix = 1 - (jaccard_index_frames[frame] / maximum)

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out assignments with a cost higher than the threshold (i.e., IoU lower than sigma_iou)
        matched_indices = [
            (r, c)
            for r, c in zip(row_ind, col_ind)
            if cost_matrix[r, c] <= (1 - sigma_iou)
        ]

        # Extract matches and their IoU values
        matches = [c for r, c in matched_indices]
        iou_values = [jaccard_index_frames[frame][r, c] for r, c in matched_indices]

        # Update tracks and jaccard values dictionaries
        tracks[frame] = matches
        jaccard_values[frame] = iou_values

    return tracks, jaccard_values
