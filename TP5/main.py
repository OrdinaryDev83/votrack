# main.py
from preprocess_data import preprocess_frames
from jaccard_index import resnet_embedding_similarity_frames
from detection_association import associate_detections_to_tracks
from track_management import track_management
from draw_and_display import draw_frames, show_video, load_images
from save_csv import save_tracking_results
import pandas as pd
from vision import load_yolo, process_frames

def process_file(images, folder_name, file_name, save_name):
    det = pd.read_csv("../Data/" + folder_name + "/" + file_name, sep=",", header=None)

    det_frames, og_len = preprocess_frames(det)

    print(f"Computing similarity matrices for {file_name} ...")
    det_jaccard_index_frames, det_similarity_frames, det_histogram_frames = resnet_embedding_similarity_frames(images, det_frames)

    print(f"Associating detections to tracks for {file_name} ...")
    det_tracks, det_jaccard_values = associate_detections_to_tracks(
        det_jaccard_index_frames,
        det_similarity_frames,
        det_histogram_frames,
        sigma_iou=0.73,
    )

    print(f"Tracking management for {file_name} ...")
    det_bboxes_for_each_frame = track_management(det_frames, det_tracks, det_jaccard_values)

    print(f"Saving tracking results for {file_name} ...")
    save_tracking_results(
        det_bboxes_for_each_frame, det_frames, save_name
    )

    print(f"Drawing frames for {file_name} ...")
    det_frame_imgs = draw_frames(og_len, images, det_bboxes_for_each_frame, det_frames)

    show_video(file_name, det_frame_imgs)

def generate_yolo_file(images):
    print("Loading yolo...")
    model_yolo = load_yolo()

    print("Generating det_yolo.txt ...")
    process_frames(model_yolo, images)

print("Loading images...")
images = load_images(525)

# generate_yolo_file(images)

# process_file(images, "det", "det.txt", "det_output.txt")
process_file(images, ".", "det_yolo.txt", "ADL-Rundle-6.txt")
process_file(images, "gt", "gt.txt", "gt_output.txt")