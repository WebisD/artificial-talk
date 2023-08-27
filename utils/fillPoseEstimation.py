import numpy as np
import pandas as pd
import pickle

column_names = ['nose_x', 'nose_y', 'left_eye_x', 'left_eye_y', 'right_eye_x', 'right_eye_y', 'left_ear_x', 'left_ear_y', 'right_ear_x', 'right_ear_y', 'left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y', 'left_elbow_x', 'left_elbow_y', 'right_elbow_x', 'right_elbow_y', 'left_wrist_x', 'left_wrist_y', 'right_wrist_x', 'right_wrist_y', 'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y', 'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y', 'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y']

def customImputer(data):
    data = np.where(data <= 0, np.nan, data)
    keypoints = data.reshape(-1,34)
    df_keypoints = pd.DataFrame(keypoints, columns=column_names)
    

    with open('models/iterative_imputer_model.pkl', 'rb') as file:
        imputed_model = pickle.load(file)
        imputed_points = imputed_model.transform(df_keypoints)
        imputed_points = imputed_points.reshape(17,2)

        return imputed_points