import numpy as np
import copy

def track_management(tracks, jaccard_values):
    bboxes_for_each_frame = {}
    frame_indices = list(tracks.keys())
    ids = {}
    new_track_id = 0
    frame_n_minus_1 = 0
    for frame_index, current_frame in enumerate(frame_indices[1:]):
        if current_frame - frame_n_minus_1 > 1:
            unmatched_tracks = []
            if frame_n_minus_1 in tracks:
                for i, track in enumerate(tracks[frame_n_minus_1]):
                    unmatched_tracks.append(track)
            ids.clear()
            bboxes_for_each_frame[current_frame] = {
                "ids": ids,
                "matches": [],
                "unmatched_tracks": unmatched_tracks,
                "unmatched_detections": [],
                "jaccard_dict": []
            }
            frame_n_minus_1 = current_frame
            continue
        # detect if we skip a frame
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
                if track in ids:
                    del ids[track]

        for i, track in enumerate(c_tracks):
            if track in p_tracks:
                if track not in ids:
                    ids[track] = new_track_id
                    new_track_id += 1
                matches.append(track)
            else:
                unmatched_detections.append(track)
                ids[track] = new_track_id
                new_track_id += 1
        
        bboxes_for_each_frame[current_frame] = {
            "ids": copy.deepcopy(ids),
            "matches": matches,
            "unmatched_tracks": unmatched_tracks,
            "unmatched_detections": unmatched_detections,
            "jaccard_dict": jaccard_dict,
        }
        frame_n_minus_1 = current_frame
    return bboxes_for_each_frame
