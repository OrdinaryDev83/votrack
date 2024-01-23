import numpy as np

sigma_iou = 0.7


def associate_detections_to_tracks(jaccard_index_frames):
    tracks = {}
    jaccard_values = {}
    for frame in jaccard_index_frames.keys():
        tracks[frame] = []
        jaccard_values[frame] = []
        for jaccard_index_frame in jaccard_index_frames[frame]:
            if len(jaccard_index_frame) == 0:
                tracks[frame].append(-1)
                continue
            max_index = np.argmax(jaccard_index_frame)
            if jaccard_index_frame[max_index] >= sigma_iou:
                tracks[frame].append(max_index)
                jaccard_values[frame].append(jaccard_index_frame[max_index])
            else:
                tracks[frame].append(-1)
                jaccard_values[frame].append(-1)
    return tracks, jaccard_values
