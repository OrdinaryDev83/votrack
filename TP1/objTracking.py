from typing import DefaultDict
from KalmanFilter import KalmanFitler
from Detector import detect
import cv2
import numpy as np
import time

kf = KalmanFitler(
        dt=1,
        u_x=0,
        u_y=0,
        std_acc=0.1,
        x_sdt_meas=0.1,
        y_sdt_meas=0.1
    )

cap = cv2.VideoCapture("randomball.avi")

frame_rate = 30
prev = 0

trajectory = []

def preprocess():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return 1
    
    to_show = frame.copy()

    detected = detect(frame)
    if detected:
        for center in detected:
            x, y = center
            # draw green circle around the detected object
            to_show = cv2.circle(to_show, (int(x), int(y)), 15, (0, 255, 0), 2)
            # Update the Kalman Filter with the current detection
            updated_state = kf.update(np.array([x, y]))
            updated_x, updated_y = int(updated_state[0]), int(updated_state[1])

            # Draw the updated position as a red rectangle
            to_show = cv2.rectangle(to_show, (int(updated_x-15), int(updated_y-15)), (int(updated_x+15), int(updated_y+15)), (0, 0, 255), 2)

            # Predict the next position
            predicted_centroid = kf.predict()
            predicted_x, predicted_y = int(predicted_centroid[0]), int(predicted_centroid[1])

            # Draw the predicted position as a blue rectangle
            to_show = cv2.rectangle(to_show, (int(predicted_x-15), int(predicted_y-15)), (int(predicted_x+15), int(predicted_y+15)), (255, 0, 0), 2)

            # Update the trajectory with the updated position
            trajectory.append((updated_x, updated_y))
            if len(trajectory) > 4:
                trajectory.pop(0)

            # Draw the trajectory
            for j in range(1, len(trajectory)):
                if trajectory[j-1] is None or trajectory[j] is None:
                    continue
                to_show = cv2.line(to_show, trajectory[j-1], trajectory[j], (0, 255, 255), 5)  # thickness is set to 5
    else:
        print("No detected")
    cv2.imshow("window", to_show)

    return 0

while cap.isOpened():
    if cv2.waitKey(1) == ord('q'):
        break

    time_elapsed = time.time() - prev
    res, image = cap.read()

    if time_elapsed > 1./frame_rate:
        prev = time.time()

        if preprocess() != 0:
            break