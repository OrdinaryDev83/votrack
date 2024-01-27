# Import necessary libraries
import cv2
from tqdm import tqdm
from drawing_functions import draw_bboxes, draw_ids, draw_trajectories
import copy

# Set the number of frames to backtrack for drawing trajectories
trajectory_motion_backtrack = 3

# Function to load images from a directory
def load_images(og_len):
    l = []
    for frame in tqdm(range(1, og_len + 1)):
        i = cv2.imread(f"../Data/img1/{frame:06d}.jpg")
        i = cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
        l.append(i)
    return l

# Function to draw bounding boxes, IDs, and trajectories on frames
def draw_frames(og_len, images_, bboxes_for_each_frame, frames):
    images = copy.deepcopy(images_)
    frame_ids = list(bboxes_for_each_frame.keys())

    # Convert images to RGB
    images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]

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

        # Draw bounding boxes and IDs for matched detections
        draw_bboxes(img, bbox_matches, (255, 255, 255))
        draw_ids(img, bbox_matches, jaccard_values, matches, (255, 255, 255))

        # Draw bounding boxes and IDs for unmatched detections
        draw_bboxes(img, bbox_unmatched_detections, (0, 255, 0))
        draw_ids(
            img,
            bbox_unmatched_detections,
            jaccard_values,
            unmatched_detections,
            (0, 255, 0),
        )

        # Draw bounding boxes and IDs for unmatched tracks
        draw_bboxes(img, bbox_unmatched_tracks, (0, 0, 255))
        draw_ids(img, bbox_unmatched_tracks, None, unmatched_tracks, (0, 0, 255))

        # Draw trajectories for matched detections
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

# Function to display the video frames
fps = 30
def show_video(file_name, frame_imgs):
    for frame in frame_imgs:
        cv2.imshow("video", frame)
        cv2.setWindowTitle("video", file_name)
        if cv2.waitKey(1000 // fps) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
