from utils.customAnnotation import Annotation
import numpy as np
import matplotlib.pyplot as plt

import openpifpaf
openpifpaf.show.Canvas.show = True
openpifpaf.show.Canvas.image_min_dpi = 50
predictor = openpifpaf.Predictor(checkpoint='shufflenetv2k16')

def transform_keypoints(points, scale=10, window_size=[800,600]):
    # Find the minimum and maximum values along each axis
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)

    # Calculate the ranges along each axis
    range_x = max_x - min_x
    range_y = max_y - min_y

    # Calculate the translation to center the points
    translate_x = (window_size[0] - scale * range_x) / 2
    translate_y = (window_size[1] - scale * range_y) / 2

    # Apply scaling and translation to the points
    transformed_points = (points - np.array([min_x, min_y])) * scale
    transformed_points += np.array([translate_x, translate_y])

    if transformed_points.max(axis=0)[0] >= window_size[0] or transformed_points.max(axis=0)[1] >= window_size[1]:
        transformed_points = transform_keypoints(points, scale=scale-1)

    return transformed_points

def show_keypoints(prediction, save_path=None, show=True, window_size=[800,600]):
    prediction = transform_keypoints(prediction, scale=10, window_size=[800,600])
    prediction = np.concatenate((prediction, np.ones((prediction.shape[0], 1))), axis=1)
    prediction = np.array(prediction)
    annotation = [Annotation(prediction)]

    width,height = window_size
    white_canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    annotation_painter = openpifpaf.show.AnnotationPainter()
    with openpifpaf.show.Canvas.image(white_canvas) as ax:
        annotation_painter.annotations(ax, annotation)
        if save_path:
            plt.savefig(save_path)
        if not show:
            plt.clf()
            plt.close()