import numpy as np

def track_management(tracks, jaccard_values):
    bboxes_for_each_frame = {}
    frame_indices = list(tracks.keys())
    for frame_index, current_frame in enumerate(frame_indices[1:]):
        previous_frame = frame_indices[frame_index - 1]
        p_tracks = np.array(tracks[previous_frame])
        c_tracks = np.array(tracks[current_frame])
        # sort
        jaccard_dict = {}
        for i, t in enumerate(c_tracks):
            jaccard_dict[t] = jaccard_values[current_frame][i]
        p_tracks = np.sort(p_tracks)
        c_tracks = np.sort(c_tracks)
        # remove -1
        p_tracks = p_tracks[p_tracks >= 0]
        c_tracks = c_tracks[c_tracks >= 0]
        # uniques only
        p_tracks = np.unique(p_tracks)
        c_tracks = np.unique(c_tracks)
        matches = []
        unmatched_tracks = []  # objects that have disappeared
        unmatched_detections = []  # new appearances of objects

        for i, track in enumerate(p_tracks):
            if track not in c_tracks:
                unmatched_tracks.append(track)

        for i, track in enumerate(c_tracks):
            if track in p_tracks:
                matches.append(track)
            else:
                unmatched_detections.append(track)
        bboxes_for_each_frame[current_frame] = {
            "matches": matches,
            "unmatched_tracks": unmatched_tracks,
            "unmatched_detections": unmatched_detections,
            "jaccard_dict": jaccard_dict,
        }
    return bboxes_for_each_frame
