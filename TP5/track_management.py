import numpy as np
from kalman_filter import KalmanFilter


def track_management(frames, tracks, jaccard_values):
    """
    Perform track management by updating the positions of objects in each frame based on the previous frames.

    Args:
        frames (dict): A dictionary containing the positions of objects in each frame.
        tracks (dict): A dictionary containing the tracks of objects in each frame.
        jaccard_values (dict): A dictionary containing the Jaccard similarity values for each object in each frame.

    Returns:
        dict: A dictionary containing the updated positions of objects in each frame, along with other information.

    """
    bboxes_for_each_frame = {}
    frame_indices = list(tracks.keys())
    kalman_filters = {}

    for frame_index, current_frame in enumerate(frame_indices[1:]):
        frame = frame_indices[frame_index]
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

        for old_matches in unmatched_tracks:
            if old_matches in kalman_filters:
                del kalman_filters[old_matches]

        for new_matches in unmatched_detections:
            kalman_filters[new_matches] = KalmanFilter(
                dt=1,
                u_x=0,
                u_y=0,
                std_acc=0.5,
                x_sdt_meas=0.5,
                y_sdt_meas=0.5,
            )
            kalman_filters[new_matches].x = np.array(
                [
                    [frames[frame][new_matches][0]],
                    [frames[frame][new_matches][1]],
                    [0],
                    [0],
                ]
            )

        for match in matches:
            if match not in kalman_filters:
                # Possibly initialize a new Kalman Filter for a new object detected
                continue

            # Predict the next state (position and velocity)
            predicted_state = kalman_filters[match].predict()

            # Extract predicted position
            x_pred, y_pred = int(predicted_state[0]), int(predicted_state[1])

            x_meas, y_meas = frames[frame][match][:2]
            kalman_filters[match].update(np.array([[x_meas], [y_meas]]))

            # Update frames with the predicted position (or measured position if update was performed)
            frames[frame][match][0] = x_pred
            frames[frame][match][1] = y_pred

        bboxes_for_each_frame[current_frame] = {
            "matches": matches,
            "unmatched_tracks": unmatched_tracks,
            "unmatched_detections": unmatched_detections,
            "jaccard_dict": jaccard_dict,
        }
    return bboxes_for_each_frame
