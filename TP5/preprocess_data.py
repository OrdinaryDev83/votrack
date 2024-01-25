# preprocess_data.py
import pandas as pd


def preprocess_frames(df):
    columns = [
        "frame",
        "id",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
        "conf",
        "x",
        "y",
        "z",
    ]
    df.columns = columns

    og_len = df["frame"].max()

    conf_threshold = 15.0
    if df["conf"].unique()[0] != 1:
        df = df[df.conf >= conf_threshold]
    
    # remove bboxes with size <= 0 when rounded
    df = df[df.bb_left.round() > 0]
    df = df[df.bb_top.round() > 0]
    df = df[df.bb_width.round() > 0]
    df = df[df.bb_height.round() > 0]

    df = df.drop(columns=["x", "y", "z"])
    return get_frames(df), og_len


def get_frames(df):
    frames = {}
    for frame in df["frame"].unique():
        current = []
        for _, row in df[df["frame"] == frame].iterrows():
            current.append(row[2:6].values)
        frames[frame] = current
    return frames
