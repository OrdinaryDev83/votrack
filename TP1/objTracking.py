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

frame_rate = 5
prev = 0

trajectory = DefaultDict(list)

def preprocess():
    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return 1
    
    to_show = frame.copy()

    detected = detect(frame)
    if detected:
        for i, center in enumerate(detected):
            x, y = center
            # red rectangle as the estimated object position
            # green circle for detected object
            # blue rectangle as the predicted object position
            # trajectory
            to_show = cv2.circle(to_show, (int(x), int(y)), 15, (0, 255, 0), 2)
            p = kf.predict()
            k_x, k_y = int(p[0]), int(p[1])
            k_v_x, k_v_y = int(p[2]), int(p[3])
            to_show = cv2.rectangle(to_show, (int(k_x-15), int(k_y-15)), (int(k_x+15), int(k_y+15)), (0, 0, 255), 2)
            p_2 = kf.update(center)
            k_x_2, k_y_2 = int(p_2[0]), int(p_2[1])
            to_show = cv2.rectangle(to_show, (int(k_x_2-15), int(k_y_2-15)), (int(k_x_2+15), int(k_y_2+15)), (255, 0, 0), 2)
            trajectory[i].append((k_x, k_y))
            if len(trajectory[i]) > 4:
                trajectory[i].pop(0)
            for j in range(1, len(trajectory[i])):
                if trajectory[i][j-1] is None or trajectory[i][j] is None:
                    continue
                # thickness is velocity
                scale = 0.2
                thickness = max(min(int(np.sqrt(k_v_x ** 2 + k_v_y ** 2) * scale), 15), 1)
                to_show = cv2.line(to_show, trajectory[i][j-1], trajectory[i][j], (0, 255, 255), thickness)
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