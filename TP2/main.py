import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

det = pd.read_csv("../Data/det/det.txt", sep=",", header=None)
gt = pd.read_csv("../Data/gt/gt.txt", sep=",", header=None)


def preprocess_frames(df):
    global og_len
    columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]
    df.columns = columns

    og_len = df["frame"].max()

    # filter bboxes with conf
    conf_threshold = 30.0
    if df["conf"].unique()[0] != 1:
        df = df[df.conf >= conf_threshold]

    df = df.drop(columns=["x", "y", "z"])

    def get_frames(df):
        frames = {}
        for frame in df["frame"].unique():
            current = []
            for _, row in df[df["frame"] == frame].iterrows():
                current.append(row[2:6].values)
            frames[frame] = current
        return frames

    return get_frames(df)


det_frames = preprocess_frames(det)
gt_frames = preprocess_frames(gt)


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


det_jaccard_index_frames = jaccard_index_frames(det_frames)
gt_jaccard_index_frames = jaccard_index_frames(gt_frames)

# associate the detections to tracks in a greedy manner using IoU threshold sigma_iou
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


det_tracks, det_jaccard_values = associate_detections_to_tracks(
    det_jaccard_index_frames
)
gt_tracks, gt_jaccard_values = associate_detections_to_tracks(gt_jaccard_index_frames)


# track management
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


det_bboxes_for_each_frame = track_management(det_tracks, det_jaccard_values)
gt_bboxes_for_each_frame = track_management(gt_tracks, gt_jaccard_values)


def draw_bboxes(frame, bboxes, color):
    for bbox in bboxes:
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ids(frame, bboxes, jaccard_dict, ids, color):
    for bbox, id in zip(bboxes, ids):
        x, y, w, h = bbox.astype(int)
        cv2.putText(frame, str(id), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2)
        if jaccard_dict is not None:
            jac = jaccard_dict[id]
            cv2.putText(
                frame,
                str(round(jac, 2)),
                (x, y + h + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                color,
                2,
            )


trajectory_motion_backtrack = 3


def draw_trajectories(frame, centroids, previous_centroids, ids, color):
    for centroid, id in zip(centroids, ids):
        # draw line between previous centroid and current centroid
        if id > len(previous_centroids) - trajectory_motion_backtrack:
            continue
        previous_centroid = previous_centroids[id]
        previous_centroid = (int(previous_centroid[0]), int(previous_centroid[1]))
        current_centroid = (int(centroid[0]), int(centroid[1]))
        cv2.arrowedLine(
            frame, previous_centroid, current_centroid, color, 3, tipLength=0.3
        )


def draw_frames(og_len, bboxes_for_each_frame, frames):
    frame_ids = list(bboxes_for_each_frame.keys())

    centroids = {}
    frame_imgs = []
    frame_index = 0
    for frame in tqdm(range(1, og_len + 1)):
        img = cv2.imread(f"../Data/img1/{frame:06d}.jpg")
        if frame not in frame_ids:
            frame_imgs.append(img)
            continue
        frame_index += 1
        matches = bboxes_for_each_frame[frame]["matches"]
        unmatched_detections = bboxes_for_each_frame[frame]["unmatched_detections"]
        unmatched_tracks = bboxes_for_each_frame[frame]["unmatched_tracks"]
        jaccard_values = bboxes_for_each_frame[frame]["jaccard_dict"]

        bbox_matches = []
        bbox_unmatched_detections = []
        bbox_unmatched_tracks = []
        for i, bbox in enumerate(frames[frame]):
            if i in matches:
                bbox_matches.append(bbox)
            elif i in unmatched_detections:
                bbox_unmatched_detections.append(bbox)
            elif i in unmatched_tracks:
                bbox_unmatched_tracks.append(bbox)

        centroids[frame] = []
        for bb in bbox_matches:
            centroids[frame].append((bb[0] + bb[2] // 2, bb[1] + bb[3] // 2))

        draw_bboxes(img, bbox_matches, (255, 255, 255))
        draw_ids(img, bbox_matches, jaccard_values, matches, (255, 255, 255))

        draw_bboxes(img, bbox_unmatched_detections, (0, 255, 0))
        draw_ids(
            img,
            bbox_unmatched_detections,
            jaccard_values,
            unmatched_detections,
            (0, 255, 0),
        )

        draw_bboxes(img, bbox_unmatched_tracks, (0, 0, 255))
        draw_ids(img, bbox_unmatched_tracks, None, unmatched_tracks, (0, 0, 255))

        if frame_index > frame_ids[trajectory_motion_backtrack]:
            previous_frame_index = frame_ids[frame_index - trajectory_motion_backtrack]
            draw_trajectories(
                img,
                centroids[frame],
                centroids[previous_frame_index],
                matches,
                (255, 255, 255),
            )

        frame_imgs.append(img)
    return frame_imgs


# pop a window with a video stream from the frame_imgs
fps = 30


def show_video(frame_imgs):
    for frame in frame_imgs:
        cv2.imshow("video", frame)
        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()


det_frame_imgs = draw_frames(og_len, det_bboxes_for_each_frame, det_frames)
gt_frame_imgs = draw_frames(og_len, gt_bboxes_for_each_frame, gt_frames)

show_video(det_frame_imgs)
show_video(gt_frame_imgs)
