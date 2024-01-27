from scipy.optimize import linear_sum_assignment
import numpy as np


def compute_score(similarity, max_similarity, histogram_score):
    """
    Compute the score for associating a detection with a track.
    
    Args:
        similarity (numpy.ndarray): Array of similarity values between detections and tracks.
        max_similarity (float): Maximum similarity value.
        histogram_score (float): Score based on histogram comparison.
    
    Returns:
        float: The computed score.
    """
    return (similarity / max_similarity) * 0.7 + histogram_score * 0.3


def associate_detections_to_tracks(
    jaccard_index_frames, similarity_frames, histogram_frames, sigma_iou
):
    """
    Associate detections to tracks based on similarity, histogram and IoU values.
    
    Args:
        jaccard_index_frames (dict): Dictionary of Jaccard index values for each frame.
        similarity_frames (dict): Dictionary of similarity values for each frame.
        histogram_frames (dict): Dictionary of histogram scores for each frame.
        sigma_iou (float): Threshold for IoU values.
    
    Returns:
        tuple: A tuple containing two dictionaries - tracks and jaccard_values.
            tracks (dict): Dictionary of tracks associated with detections for each frame.
            jaccard_values (dict): Dictionary of IoU values for each associated detection.
    """
    tracks = {}
    jaccard_values = {}

    # find the maximum value in the keys (which are lists) of similarity_frames
    maximum = np.max(
        [np.max(similarity_frames[key]) for key in similarity_frames.keys()]
    )
    for frame in jaccard_index_frames.keys():
        # Convert IoU to cost matrix (1 - IoU)
        score_matrix = compute_score(
            similarity_frames[frame], maximum, histogram_frames[frame]
        )
        cost_matrix = 1 - score_matrix

        # Apply Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # Filter out assignments with a cost higher than the threshold (i.e., IoU lower than sigma_iou)
        matched_indices = [
            (r, c)
            for r, c in zip(row_ind, col_ind)
            if jaccard_index_frames[frame][r, c] >= sigma_iou
            and score_matrix[r, c] >= 0.5
        ]

        # Extract matches and their IoU values
        matches = [c for r, c in matched_indices]
        iou_values = [score_matrix[r, c] for r, c in matched_indices]

        # Update tracks and jaccard values dictionaries
        tracks[frame] = matches
        jaccard_values[frame] = iou_values

    return tracks, jaccard_values
