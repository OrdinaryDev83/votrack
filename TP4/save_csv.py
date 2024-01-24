def save_tracking_results(gt_bboxes_for_each_frame, frames, output_filename):
    with open(output_filename, "w") as file:
        for frame in sorted(gt_bboxes_for_each_frame.keys()):
            data = gt_bboxes_for_each_frame[frame]
            matched_tracks = data["matches"]
            bbox_data = frames[frame]

            for track_id in matched_tracks:
                if track_id == -1 or track_id >= len(bbox_data):  # Skip unmatched tracks
                    continue

                bb_left, bb_top, bb_width, bb_height = bbox_data[track_id]
                # Write the formatted line to the file
                file.write(
                    f"{frame},{track_id},{bb_left},{bb_top},{bb_width},{bb_height},1,-1,-1,-1\n"
                )
