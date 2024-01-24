import cv2
from tqdm import tqdm
from drawing_functions import draw_bboxes, draw_ids, draw_trajectories

trajectory_motion_backtrack = 3

def load_images(og_len):
    l = []
    for frame in tqdm(range(1, og_len + 1)):
        l.append(cv2.imread(f"../Data/img1/{frame:06d}.jpg"))
    return l

def draw_frames(og_len, images, bboxes_for_each_frame, frames):
    frame_ids = list(bboxes_for_each_frame.keys())

    centroids = {}
    frame_imgs = []
    frame_index = 0
    for frame in tqdm(range(og_len)):
        img = images[frame]
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
                trajectory_motion_backtrack,
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
