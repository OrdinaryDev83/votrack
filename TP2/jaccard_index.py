import numpy as np


# compute the Jaccard index between two bounding boxes
def jaccard_index(a, b):
    a_left, a_top, a_width, a_height = a
    b_left, b_top, b_width, b_height = b
    a_right = a_left + a_width
    a_bottom = a_top + a_height
    b_right = b_left + b_width
    b_bottom = b_top + b_height
    xA = max(a_left, b_left)
    yA = max(a_top, b_top)
    xB = min(a_right, b_right)
    yB = min(a_bottom, b_bottom)
    interArea = max(0, xB - xA) * max(0, yB - yA)
    aArea = a_width * a_height
    bArea = b_width * b_height
    return interArea / float(aArea + bArea - interArea)


def jaccard_index_frames(frames):
    jaccard_index_frames = {}
    frame_indices = list(frames.keys())
    for frame_index, frame in enumerate(frames):
        if frame == 1:
            continue
        jaccard_index_frame = np.zeros(
            (len(frames[frame]), len(frames[frame_indices[frame_index - 1]]))
        )
        for i, a in enumerate(frames[frame]):
            for j, b in enumerate(frames[frame_indices[frame_index - 1]]):
                jaccard_index_frame[i][j] = jaccard_index(a, b)
        jaccard_index_frames[frame] = jaccard_index_frame
    return jaccard_index_frames
