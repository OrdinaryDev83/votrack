# main.py
from preprocess_data import preprocess_frames
from jaccard_index import jaccard_index_frames
from detection_association import associate_detections_to_tracks
from track_management import track_management
from draw_and_display import draw_frames, show_video
import pandas as pd

det = pd.read_csv("../Data/det/det.txt", sep=",", header=None)
gt = pd.read_csv("../Data/gt/gt.txt", sep=",", header=None)

trajectory_motion_backtrack = 3

det_frames, og_len = preprocess_frames(det)
gt_frames, _ = preprocess_frames(gt)

det_jaccard_index_frames = jaccard_index_frames(det_frames)
gt_jaccard_index_frames = jaccard_index_frames(gt_frames)

det_tracks, det_jaccard_values = associate_detections_to_tracks(det_jaccard_index_frames)
gt_tracks, gt_jaccard_values = associate_detections_to_tracks(gt_jaccard_index_frames)

det_bboxes_for_each_frame = track_management(det_tracks, det_jaccard_values)
gt_bboxes_for_each_frame = track_management(gt_tracks, gt_jaccard_values)

det_frame_imgs = draw_frames(og_len, det_bboxes_for_each_frame, det_frames)
gt_frame_imgs = draw_frames(og_len, gt_bboxes_for_each_frame, gt_frames)

show_video(det_frame_imgs)
show_video(gt_frame_imgs)
