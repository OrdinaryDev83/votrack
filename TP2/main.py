import numpy as np
import pandas as pd
import cv2

def load_det():
    # Parse and load the det.txt
    det = pd.read_csv('det.txt', sep=',', header=None)