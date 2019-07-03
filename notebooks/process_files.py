import argparse
import json
import shutil
import os
import matplotlib.pyplot as plt
from itertools import chain
from pathlib import Path, PosixPath
from typing import *

import cv2
import numpy as np
import pytesseract
import skimage.morphology
import tqdm
from scipy import optimize

from constants import *


def initialize_env(output_dir: PosixPath):
    if output_dir.exists():
        shutil.rmtree(output_dir.as_posix())
    output_dir.mkdir(parents=True)
    (output_dir / UNKNOWN_FOLDER_NAME).mkdir()
    (output_dir / "F0").mkdir()
    (output_dir / "F4").mkdir()


def is_img_unknown(img_path: PosixPath) -> bool:
    img = cv2.imread(img_path.as_posix())
    img -= img.min()
    img = img[
          Y_CHECK_DUAL_IMAGES_RECTANGLE:Y_CHECK_DUAL_IMAGES_RECTANGLE + H_CHECK_DUAL_IMAGES_RECTANGLE,
          X_CHECK_DUAL_IMAGES_RECTANGLE:X_CHECK_DUAL_IMAGES_RECTANGLE + W_CHECK_DUAL_IMAGES_RECTANGLE]
    total_values_in_area = img.sum()
    return total_values_in_area > 0


def split_image(whole_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    without_bar = whole_img[:-LOWER_BAR_HEIGHT]
    height = without_bar.shape[0]
    up = without_bar[:height // 2]
    down = without_bar[height // 2:]
    return down, up


def get_variance_map_between_areas(lower_half: np.ndarray, upper_half: np.ndarray) -> np.ndarray:
    diff = np.abs(lower_half - upper_half)
    variance = np.var(diff, axis=-1)
    var_img = (variance > VARIANCE_THRESHOLD).astype(np.uint8) * 255
    var_img = cv2.medianBlur(var_img, 5)
    return var_img


def get_absolute_parameters_of_radial_polar_area(lower_half: np.ndarray,
                                                 upper_half: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    var_img = get_variance_map_between_areas(lower_half, upper_half)
    cnt = cv2.findContours(var_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = [(c, cv2.contourArea(c)) for c in cnt[0]]

    max_contour = max(contours, key=lambda single_contour: single_contour[1])[0]

    bbox = cv2.boundingRect(max_contour)

    y = bbox[1] - PADDING_AROUND_RADIAL_POLAR_FRAGMENT
    h = bbox[3] + PADDING_AROUND_RADIAL_POLAR_FRAGMENT * 2

    x = bbox[0] - PADDING_AROUND_RADIAL_POLAR_FRAGMENT
    w = bbox[2] + PADDING_AROUND_RADIAL_POLAR_FRAGMENT * 2

    cut_area = lower_half[y:y + h, x:x + w]

    rectangle_params = [
        int(x), int(y), int(w), int(h)
    ]
    return cut_area, rectangle_params


def circle_mask_fitness_function(params, *args):
    """
    Function takes params of a circle and matches to to current img. It does so by
    making 1px wide border around the circle and then taking intersection of predicted
    mask and those found in the `img_back`.
    params:
        params: params to minimze (x_center, y_center, radius) of the circle
    returns:
        Scalar value of degree which mask and circle matches.
    """
    img = args[0]
    height, width = img.shape[:2]
    params = [int(val) for val in params]
    x_c, y_c, r = params
    
    predicted = np.zeros_like(img).astype(np.uint8)
    predicted = cv2.circle(predicted, (x_c, y_c), r, 255, 1).get()
    predicted //= 255
    
    penalty = 0
    condition = (
            x_c + r >= width
            or x_c - r < 0
            or y_c + r >= height
            or y_c - r < 0
    )
    if condition:
        penalty = + 1e8

    fitness_value = np.sum((predicted > 0) & (img > 0))
    return -fitness_value + penalty


def perform_multiple_fft_clearing_with_thresholding(input_image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = 1 - laplacian / laplacian.max()

    f = np.fft.fft2(laplacian)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.abs(fshift)
    mask = magnitude_spectrum > np.percentile(magnitude_spectrum, 99)

    fshift[mask] = 0
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift).real
    img_back = np.abs(img_back)

    img_back = (img_back - img_back.min()) / (img_back.max() - img_back.min())
    img_back = (img_back > 0.5).astype(np.uint8)
    return img_back


def get_circle_absolute_center_radius(radial_polar_rectangle_area: np.ndarray,
                                      radial_polar_rectangle_parameters: List[int]) -> List[int]:
    img = perform_multiple_fft_clearing_with_thresholding(radial_polar_rectangle_area)

    ret = optimize.differential_evolution(
        circle_mask_fitness_function,
        bounds=[(0, img.shape[1]),
                (0, img.shape[0]),
                (1, max(img.shape[0] // 2, img.shape[1] // 2))],
        args=(img,),
        maxiter=FITNESS_NUMBER_OF_ITERATIONS,
        popsize=FITNESS_POPULATION_SIZE,
        tol=0.001
    )
    x_center, y_center, radius = [int(val) for val in ret.x]

    circle_params = [
        int(radial_polar_rectangle_parameters[0] + x_center),
        int(radial_polar_rectangle_parameters[1] + y_center),
        int(radius)
    ]
    return circle_params


def get_masked_out_circle(lower_half: np.ndarray, circle_absolute_parameters: List[int]) -> np.ndarray:
    mask = np.zeros(lower_half.shape[:-1]).astype(np.uint8)
    height, width = mask.shape
    x_center, y_center, radius = circle_absolute_parameters

    radius -= 2
    cv2.circle(mask, (x_center, y_center), radius, 255, -1)
    x_min = max(x_center - radius, 0)
    x_max = min(x_center + radius, width)
    y_min = max(y_center - radius, 0)
    y_max = min(y_center + radius, height)

    mask = mask == 255
    output_image = lower_half.copy()
    output_image[~mask] = 0
    output_image = output_image[y_min:y_max, x_min:x_max]
    return output_image


def get_legends(upper_half: np.ndarray) -> Tuple[np.ndarray, ...]:
    left_side = upper_half[:, :END_OF_LEFT_SIDE_LEGEND_X]
    right_side = upper_half[:, START_OF_RIGHT_SIDE_LEGEND_X:]

    lower_right_side = right_side[END_OF_COLOR_BAR_LEGEND_Y:]
    upper_right_side = right_side[:END_OF_COLOR_BAR_LEGEND_Y]

    color_bar_legend = upper_right_side[:, :END_OF_COLOR_BAR_LEGEND_X]
    color_bar_legend_range = upper_right_side[:, END_OF_COLOR_BAR_LEGEND_X:]

    return left_side, right_side, lower_right_side, color_bar_legend, color_bar_legend_range


def get_text_for_img(image_with_text: np.ndarray, relative_x_of_analysed_area: int,
                     relative_y_of_analysed_area: int) -> List[Tuple[str, List[int]]]:
    height, width = image_with_text.shape[:-1]
    gray = cv2.cvtColor(image_with_text, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    skeleton = skimage.morphology.skeletonize(gray // 255).astype(np.uint8) * 255
    text_blocks = skeleton.copy()

    text_blocks[:] = skeleton.max(axis=1, keepdims=True)

    cnts, _ = cv2.findContours(text_blocks, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    lines = []

    for cnt in cnts:
        x_min = max(np.min(cnt[..., 0]) - 3, 0)
        x_max = min(np.max(cnt[..., 0]) + 3, width)
        y_min = max(np.min(cnt[..., 1]) - 3, 0)
        y_max = min(np.max(cnt[..., 1]) + 3, height)
        cut = image_with_text[y_min:y_max, x_min:x_max]
        lines.append((cut, (x_min, y_min, x_max - x_min, y_max - y_min)))

    text_with_coordinates = []

    for i, line in enumerate(lines):
        line_img, line_coords = line
        cut = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
        cut = cv2.resize(cut, None, fx=4, fy=4, interpolation=cv2.INTER_LANCZOS4)
        cut = 255 - (cut > 127).astype(np.uint8) * 255
        text = pytesseract.image_to_string(cut, config=TESSERACT_CONFIG)
        for to_replace in STRINGS_TO_CLEAR_OUT:
            text = text.replace(to_replace, "")

        x, y, w, h = line_coords
        text_with_coordinates.append(
            (text, (x + relative_x_of_analysed_area, y + relative_y_of_analysed_area, w, h))
        )

    text_with_coordinates = list(sorted(text_with_coordinates,
                                        key=lambda text_params: text_params[1][1]))

    return text_with_coordinates


def annotation_data_to_string(annotation_data: List[Tuple[str, List[int]]]) -> str:
    output_data = []
    for text, (x, y, w, h) in annotation_data:
        output_data.append(
            {
                "text": text,
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h)
            }
        )
    return json.dumps(output_data, indent=4)


def get_annotations(upper_half: np.ndarray) -> List[str]:
    left_side, right_side, lower_right_side, color_bar_legend, color_bar_legend_range = get_legends(upper_half)

    left_side_annotation_data = get_text_for_img(left_side, 0, 0)
    legend_bar_annotation_data = get_text_for_img(color_bar_legend_range,
                                                  START_OF_RIGHT_SIDE_LEGEND_X + END_OF_COLOR_BAR_LEGEND_X,
                                                  0)
    lower_right_annotation_data = get_text_for_img(lower_right_side,
                                                   START_OF_RIGHT_SIDE_LEGEND_X,
                                                   END_OF_COLOR_BAR_LEGEND_Y)

    left_side_annotation = annotation_data_to_string(left_side_annotation_data)
    legend_bar_annotation = annotation_data_to_string(legend_bar_annotation_data)
    lower_right_annotation = annotation_data_to_string(lower_right_annotation_data)
    return [
        left_side_annotation, legend_bar_annotation, lower_right_annotation
    ]


def process_single_image_path(
    img_path: PosixPath, 
    output_dir: PosixPath
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, str, str, str
]:
    an_img = cv2.imread(img_path.as_posix())

    lower, upper = split_image(an_img)
    radial_polar_area, radial_polar_coordinates = get_absolute_parameters_of_radial_polar_area(lower, upper)
    circle_coordinates = get_circle_absolute_center_radius(radial_polar_area, radial_polar_coordinates)
    masked_circle = get_masked_out_circle(lower, circle_coordinates)
    left_side_annotation, legend_bar_annotation, lower_right_annotation = get_annotations(upper)

    coordinates = {
        "radial_polar_area": {
            "x": radial_polar_coordinates[0],
            "y": radial_polar_coordinates[1],
            "w": radial_polar_coordinates[2],
            "h": radial_polar_coordinates[3]
        },

        "circle": {
            "x": circle_coordinates[0],
            "y": circle_coordinates[1],
            "radius": circle_coordinates[2]
        }
    }

    coordinates_str = json.dumps(coordinates, indent=4)

    folder_name = img_path.with_suffix("").name
    folder_path = output_dir / folder_name
    folder_path.mkdir(exist_ok=True)

    cv2.imwrite((folder_path / "upper.png").as_posix(), upper)
    cv2.imwrite((folder_path / "lower.png").as_posix(), lower)
    cv2.imwrite((folder_path / "radial_polar_area.png").as_posix(), radial_polar_area)
    cv2.imwrite((folder_path / "circle.png").as_posix(), masked_circle)

    with open((folder_path / "coordinates.json").as_posix(), "w") as f:
        f.write(coordinates_str)

    with open((folder_path / "left_side_annotation.json").as_posix(), "w") as f:
        f.write(left_side_annotation)

    with open((folder_path / "legend_bar_annotation.json").as_posix(), "w") as f:
        f.write(legend_bar_annotation)

    with open((folder_path / "lower_right_annotation.json").as_posix(), "w") as f:
        f.write(lower_right_annotation)
        
    return lower, upper, radial_polar_area, masked_circle, coordinates, left_side_annotation, legend_bar_annotation, lower_right_annotation


def process_all(input_dir: PosixPath, output_dir: PosixPath):
    tifs = list(chain(
        (input_dir / "F0").rglob("*.tif"),
        (input_dir / "F4").rglob("*.tif")
    ))

    before = len(tifs)
    tifs = [
        tif for tif in tifs
        if not tif.name.startswith(".")
    ]

    print("Total unreadable files: {}".format(before - len(tifs)))
    print("Images to process: {}".format(len(tifs)))

    total_unknowns, total_normal = 0, 0
    errors = []
    for filename in tqdm.tqdm(tifs):
        if is_img_unknown(filename):
            total_unknowns += 1
            target_dir = output_dir / UNKNOWN_FOLDER_NAME
            shutil.copy2(filename.as_posix(), target_dir.as_posix())
            continue

        try:
            target_dir = output_dir / filename.parent.name
            process_single_image_path(filename, target_dir)
            total_normal += 1

        except Exception as e:
            print(filename, e)
            errors.append(": ".join([filename.as_posix(), str(e)]))
            target_dir = output_dir / UNKNOWN_FOLDER_NAME
            shutil.copy2(filename.as_posix(), target_dir.as_posix())

    with open("error_files.txt", "w") as f:
        f.write("\n".join(errors))

    print("Total normal images: {}, total unknowns: {}".format(total_normal, total_unknowns))
    print("It is accordingly: {:.4f}% and {:.4f}%".format(
        total_normal / len(tifs) * 100,
        total_unknowns / len(tifs) * 100
    ))


def refine_files(input_dir: PosixPath, output_dir: PosixPath):
    print("Files to refine: {}".format(len(ENFORCE_FILENAMES)))
    tifs = list(chain(
        (input_dir / "F0").rglob("*.tif"),
        (input_dir / "F4").rglob("*.tif")
    ))

    tifs = [tif for tif in tifs if tif.name in ENFORCE_FILENAMES]
    assert len(tifs) == len(ENFORCE_FILENAMES)
    errors = []

    for filename in tqdm.tqdm(tifs):
        try:
            target_dir = output_dir / filename.parent.name
            process_single_image_path(filename, target_dir)
            os.remove((output_dir / UNKNOWN_FOLDER_NAME / filename.name).as_posix())

        except Exception as e:
            print(filename, e)
            errors.append(": ".join([filename.as_posix(), str(e)]))

    with open("refinement_error_files.txt", "w") as f:
        f.write("\n".join(errors))


def main():
    parser = argparse.ArgumentParser(
        description="Script for processing usg data"
    )

    parser.add_argument("input_dir", help="Input usg directory with F0 and F4")
    parser.add_argument("output_dir", help="General output directory")

    args = parser.parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    initialize_env(output_dir)
    process_all(input_dir, output_dir)
    refine_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
