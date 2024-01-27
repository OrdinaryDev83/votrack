# preprocess_data.py
import pandas as pd


def preprocess_frames(df):
    """
    Preprocesses the given DataFrame containing frame data.

    Args:
        df (pandas.DataFrame): The DataFrame containing frame data.

    Returns:
        tuple: A tuple containing two elements:
            - The preprocessed DataFrame.
            - The original length of the "frame" column in the DataFrame.
    """
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
    """
    Extracts frames from a DataFrame and returns a dictionary of frames.

    Args:
        df (DataFrame): The input DataFrame containing the frames.

    Returns:
        dict: A dictionary where the keys are the frame numbers and the values are lists of frame data.

    """
    frames = {}
    for frame in df["frame"].unique():
        current = []
        for _, row in df[df["frame"] == frame].iterrows():
            current.append(row[2:6].values)
        frames[frame] = current
    return frames
