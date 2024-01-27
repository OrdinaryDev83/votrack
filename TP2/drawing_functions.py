import cv2


def draw_bboxes(frame, bboxes, color):
    for bbox in bboxes:
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ids(frame, ids, bboxes, jaccard_dict, matches, color):
    for bbox, id in zip(bboxes, matches):
        x, y, w, h = bbox.astype(int)
        if id in ids:
            cv2.putText(
                frame, str(ids[id]), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2
            )
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


def draw_trajectories(
    frame, centroids, previous_centroids, ids, color, trajectory_motion_backtrack
):
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
