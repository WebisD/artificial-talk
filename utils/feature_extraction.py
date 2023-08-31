import numpy as np
import pandas as pd

COCO_Person_Keypoints = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
]

desired_angles = [
        ["left_wrist", "left_elbow", "left_shoulder"], ["right_wrist", "right_elbow", "right_shoulder"], 
        ["left_elbow", "left_shoulder", "left_hip"], ["right_elbow", "right_shoulder", "right_hip"],
        ["left_knee", "left_ankle", "left_hip"], ["right_knee", "right_ankle", "right_hip"]
    ]

new_columns = ["left_elbow", "right_elbow", "left_shoulder", "right_shoulder", "left_knee", "right_knee"]

def transform_data(df):
    grouped_keypoints = pd.DataFrame()

    for new_col in COCO_Person_Keypoints:
        x_col = f'{new_col}_x'
        y_col = f'{new_col}_y'
        grouped_keypoints[new_col] = df[[x_col, y_col]].apply(np.array, axis=1)

    result_df = pd.DataFrame()

    for new_col, group in zip(new_columns, desired_angles):
        group_data = grouped_keypoints[group].apply(calculate_angle, axis=1)
        result_df[new_col] = group_data
    
    return result_df

def calculate_angle(points):
    p_line1, p_common, p_line2 = points

    vector1 = np.array(p_line1) - np.array(p_common)
    vector2 = np.array(p_line2) - np.array(p_common)
    
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)
    cos_angle = dot_product / (magnitude1 * magnitude2)
    
    angle_radians = np.arccos(cos_angle)
    angle_degrees = np.degrees(angle_radians)
    
    # Determinando a orientação das retas
    cross_product = np.cross(vector1, vector2)
    
    if cross_product < 0:
        angle_degrees = -angle_degrees
    
    return angle_degrees
