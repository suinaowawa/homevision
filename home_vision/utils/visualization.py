"""HomeVision visualization utils"""
# pylint: disable=invalid-name
import colorsys
from typing import List

import cv2
import numpy as np


def gaussian(
    image: np.ndarray,
    mean: np.ndarray,
    covariance: np.ndarray,
    color: tuple,
    label: str=None
):
    """Draw 95% confidence ellipse of a 2-D Gaussian distribution.
    Parameters
    ----------
    mean : array_like
        The mean vector of the Gaussian distribution (ndim=1).
    covariance : array_like
        The 2x2 covariance matrix of the Gaussian distribution.
    label : Optional[str]
        A text label that is placed at the center of the ellipse.
    """
    # chi2inv(0.95, 2) = 5.9915
    vals, vecs = np.linalg.eigh(5.9915 * covariance)
    indices = vals.argsort()[::-1]
    vals, vecs = np.sqrt(vals[indices]), vecs[:, indices]

    center = int(mean[0] + .5), int(mean[1] + .5)
    axes = int(vals[0] + .5), int(vals[1] + .5)
    angle = int(180. * np.arctan2(vecs[1, 0], vecs[0, 0]) / np.pi)
    cv2.ellipse(
        image, center, axes, angle, 0, 360, color, 2)
    if label is not None:
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)

def rectangle(
    image: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    color: tuple,
    thickness: float,
    label: str=None
):
    """Draw a rectangle.
    Parameters
    ----------
    x : float | int
        Top left corner of the rectangle (x-axis).
    y : float | int
        Top let corner of the rectangle (y-axis).
    w : float | int
        Width of the rectangle.
    h : float | int
        Height of the rectangle.
    label : Optional[str]
        A text label that is placed at the top left corner of the
        rectangle.
    """
    pt1 = int(x), int(y)
    pt2 = int(x + w), int(y + h)
    cv2.rectangle(image, pt1, pt2, color, thickness)
    if label is not None:
        text_size = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_PLAIN, 1, thickness)

        center = pt1[0] + 5, pt1[1] + 5 + text_size[0][1]
        pt2 = pt1[0] + 10 + text_size[0][0], pt1[1] + 10 + \
            text_size[0][1]
        cv2.rectangle(image, pt1, pt2, color, -1)
        cv2.putText(image, label, center, cv2.FONT_HERSHEY_PLAIN,
                    1, (255, 255, 255), thickness)


def create_unique_color_float(tag: int, hue_step: float=0.41) -> tuple:
    """Create a unique RGB color code for a given track id (tag).
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.
    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).
    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]
    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b


def create_unique_color_uchar(tag: int, hue_step: float=0.41) -> tuple:
    """Create a unique RGB color code for a given track id (tag).
    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.
    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).
    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]
    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return int(255*r), int(255*g), int(255*b)


FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)  # BGR, white
MSGCOLOR = (128, 128, 128)  # BGR, gray
THICKNESS = 2
LINETYPE = cv2.LINE_AA

def hex2color(h):
    """Convert the 6-digit hex string to tuple of 3 int value (RGB)"""
    return (int(h[:2], 16), int(h[2:4], 16), int(h[4:], 16))

plate_blue = '03045e-023e8a-0077b6-0096c7-00b4d8-48cae4'
plate_blue = plate_blue.split('-')
plate_blue = [hex2color(h) for h in plate_blue]
PLATE = [x[::-1] for x in plate_blue]

def draw_actions(
    image: np.ndarray, actions: List, actions_scores: List, bboxes: List
) -> np.ndarray:
    """draw the actions of person on image"""
    overlay = image.copy()
    for k, bbox in enumerate(bboxes):
        xmin, ymin, xmax, ymax = bbox
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), PLATE[0], 2)
        try:
            person_actions = actions[k]
            person_action_scores = actions_scores[k]
        except IndexError:
            person_actions = None
            person_action_scores = None

        if person_actions is not None:
            for i, action in enumerate(person_actions):
                score = str(person_action_scores[i])
                text = ' '.join([action, score])
                textsize = cv2.getTextSize(text, FONTFACE, FONTSCALE,
                                            THICKNESS)[0]
                textwidth = textsize[0]
                textheight = textsize[1]+8
                location = (0 + int(xmin), textheight + i * textheight + int(ymin))

                diag0 = (location[0] + textwidth, location[1] - textheight+4)
                diag1 = (location[0], location[1] + 2)
                if 'fall' in text.lower():
                    cv2.rectangle(overlay, diag0, diag1, (0, 0, 255), -1)
                else:
                    cv2.rectangle(overlay, diag0, diag1, PLATE[i + 1], -1)

                cv2.putText(overlay, text, location, FONTFACE, FONTSCALE,
                            FONTCOLOR, THICKNESS, LINETYPE)
    alpha = 0.8
    image_new = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return image_new
