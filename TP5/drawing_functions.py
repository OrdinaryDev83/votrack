import cv2


def draw_bboxes(frame, bboxes, color):
    """
    Draw bounding boxes on the given frame.

    Args:
        frame: The frame to draw on.
        bboxes: List of bounding boxes.
        color: Color of the bounding boxes.

    Returns:
        None
    """
    for bbox in bboxes:
        x, y, w, h = bbox.astype(int)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


def draw_ids(frame, bboxes, jaccard_dict, ids, color):
    """
    Draw IDs and Jaccard scores on the given frame.

    Args:
        frame: The frame to draw on.
        bboxes: List of bounding boxes.
        jaccard_dict: Dictionary of Jaccard scores.
        ids: List of IDs.
        color: Color of the text.

    Returns:
        None
    """
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


def draw_trajectories(
    frame, centroids, previous_centroids, ids, color, trajectory_motion_backtrack
):
    """
    Draw trajectories on the given frame.

    Args:
        frame: The frame to draw on.
        centroids: List of current centroids.
        previous_centroids: List of previous centroids.
        ids: List of IDs.
        color: Color of the trajectories.
        trajectory_motion_backtrack: Number of previous centroids to consider.

    Returns:
        None
    """
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
