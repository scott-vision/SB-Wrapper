from ultralytics import YOLO
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import sys
import contextlib
import os
import math
import argparse
from datetime import datetime
import time




# Function to find the next experiment folder name and create it
def create_exp_folder(base_dir):
    """
    Creates a new experiment folder based on existing folders in a base directory.

    Parameters:
    - base_dir: Base directory where experiment folders are created.

    This function creates a new folder with an incremented "exp###" pattern name in the base_dir.
    The pattern starts from "exp001" and increments by 1 based on the highest numbered folder found.
    """
    global experiment_path
    # Get all items in the base directory
    items = os.listdir(base_dir)

    # Filter out items that are not directories or don't match the 'exp###' pattern
    exp_folders = [item for item in items if item.startswith('exp') and item[3:].isdigit() and os.path.isdir(os.path.join(base_dir, item))]

    # Sort the folders to find the highest number
    exp_folders.sort()

    # Determine the next experiment number
    if exp_folders:
        # Extract the highest current number and add 1
        last_exp_num = int(exp_folders[-1][3:])
        next_exp_num = last_exp_num + 1
    else:
        # If no such folder exists, start with 1
        next_exp_num = 1

    # Format the new folder name
    new_folder_name = f"exp{next_exp_num:03d}"

    # Create the new experiment folder
    experiment_path = os.path.join(base_dir, new_folder_name)
    os.makedirs(experiment_path, exist_ok=True)


def get_outimg_path():
    """
    Generates a new filename for saving temporary images within an experiment folder.

    This function generates a new filename in the form "tmpimg###.png" based on the highest
    numbered image found in the current experiment folder. If no images are present, numbering
    starts from "tmpimg001.png".

    Returns:
    - img_path: Full path to the new image file.
    """
    global experiment_path
    # Get all items in the base directory
    items = os.listdir(experiment_path)

    # Filter out items that are not directories or don't match the 'tmpimg###' pattern
    tmp_images = [item for item in items if item.startswith('tmpimg') and item[-7:-4].isdigit() and os.path.isfile(os.path.join(experiment_path, item))]

    # Sort the folders to find the highest number
    tmp_images.sort()

    # Determine the next experiment number
    if tmp_images:
        # Extract the highest current number and add 1
        last_img_num = int(tmp_images[-1][-7:-4])
        next_img_num = last_img_num + 1
    else:
        # If no such folder exists, start with 1
        next_img_num = 1

    # Format the new folder name
    new_file_name = f"tmpimg{next_img_num:03d}.png"

    # Create the new experiment folder
    img_path = os.path.join(experiment_path, new_file_name)
    return img_path


def filter_and_sort_detections(detections, class_info):
    """
    Filters and sorts object detection results based on class-specific confidence
    levels and importance rankings.

    Parameters:
    - detections: List of detection results in the format [cls, x, y, w, h, confidence].
    - class_info: Dictionary with class id as keys and tuples as values, where each tuple
                  contains (class name, acceptable confidence level, ranking).

    Returns:
    - A list of filtered and sorted detections.
    """
    # Filter detections by acceptable confidence level
    filtered_detections = [
        det for det in detections
        if det[5] >= class_info[det[0]][1] and class_info[det[0]][2] != -1
    ]

    # Sort detections by ranking and then by confidence (descending)
    sorted_detections = sorted(
        filtered_detections,
        key=lambda det: (class_info[det[0]][2], -det[5])
    )

    return sorted_detections

def global_filter_and_sort_detections(all_detections, class_info):
    """
    Filters and sorts all detections across the entire dataset.

    Parameters:
    - all_detections: List of all detections with their details.
    - class_info: Dictionary with class-specific configuration.

    Returns:
    - List of globally filtered and sorted detections.
    """
    # Apply global filter based on class-specific confidence thresholds
    filtered_detections = [
        det for det in all_detections if det[3] >= class_info[det[4]][1]
    ]

    # Sort globally by class ranking and confidence
    sorted_detections = sorted(
        filtered_detections, key=lambda det: (class_info[det[4]][2], -det[3])
    )
    return sorted_detections


def plot_image_with_results(image, boxes, class_names, class_info, file_path):
    """
    Plots an image with bounding boxes and class:confidence labels.

    Parameters:
    - image: 2D array representing the image.
    - boxes: List of lists, where each inner list contains:
             [class, x, y, w, h, confidence]
             Coordinates and sizes are not normalized.
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image, cmap='gray')  # Assuming the image is in grayscale
    ax.axis('off')  # Turn off the axis

    filtered_boxes = filter_and_sort_detections(boxes, class_info)


    for box in filtered_boxes:
        class_id, x, y, w, h, confidence = box
        # Create a Rectangle patch
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')

        # Add the rectangle to the plot
        ax.add_patch(rect)

        # Add label
        label = f"{class_names[class_id]}:{confidence:.2f}"
        ax.text(x, y, label, color='white', fontsize=8, ha='left', va='bottom',
                bbox=dict(boxstyle="square,pad=0.1", fc="black", ec="none", alpha=0.5))

    
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to prevent it from displaying in the notebook



def preprocess_image(img):
    """
    Preprocesses the input image by applying a percentile-based threshold and normalizing pixel values.

    Parameters:
    - img (array): Input image array, assumed to be 2D or 3D with identical channels.

    Returns:
    - img (array): Preprocessed image normalized to a 0-255 range and converted to uint8 type.
    """

    percentile = 0.03
    percentile_threshold = 100 - percentile  # 100 - 0.3
    threshold_value = np.percentile(img, percentile_threshold)
    # we assume 2D here so if its 3D at this point the 3rd index we assume is channels that are identical as others for saving
    if len(img.shape) == 3:
        h, w, _ = img.shape
        timepoint_data = img[:,:, 0]
    elif len(img.shape) == 2:
        h, w = img.shape
        timepoint_data = img
    else:
        print("Shape Error in predict.preprocess_image")

    timepoint_data_normalized = np.where(timepoint_data > threshold_value, threshold_value, timepoint_data)

    # Normalize the image to the range of 0 to 1

    timepoint_data_normalized2 = (timepoint_data_normalized - timepoint_data_normalized.min()) / (
        timepoint_data_normalized.max() - timepoint_data_normalized.min())
    # plt.imshow(timepoint_data_normalized2)

    img = (timepoint_data_normalized2 * 255).astype(np.uint8)

    return img

def split_image(img):
    """
    Splits a image into multiple 640x640 tiles for processing.

    Parameters:
    - img: Input 2D numpy array representing the grayscale image.

    Returns:
    - List of tuples, where each tuple contains:
        (split_image, top_left_x, top_left_y) representing each split image.
    """
    height, width = img.shape

    desired_im_size = 640
    split_images = []
    for row in range(math.ceil(height/desired_im_size)):  # Two rows
        for col in range(math.ceil(width/desired_im_size)):  # Three columns
            # initial topLeft
            x1 = col * desired_im_size
            y1 = row * desired_im_size
            # calculate and limit bottomRight
            x2 = min(x1+desired_im_size, width)
            y2 = min(y1+desired_im_size, height)
            # recalculate topLeft
            x1 = x2 - desired_im_size
            y1 = y2 - desired_im_size
            # get sub image
            split_img = img[y1:y2, x1:x2]
            split_images.append((split_img, x1, y1))

    return split_images

def adjust_coordinates(detections, x_offset, y_offset):
    """
    Adjusts the detection coordinates to the original image's coordinate system.

    Parameters:
    - detections (list): List of detection objects containing bounding box coordinates.
    - x_offset (float): Horizontal offset to adjust the coordinates relative to the original image.
    - y_offset (float): Vertical offset to adjust the coordinates relative to the original image.

    Returns:
    - detections_adjusted (list): List of adjusted detections with coordinates relative to the original image.
    """
    detections_adjusted=[]
    for detection in detections:
        box =[]
        box.append(int(detection.cls))
        box.append(float(detection.xyxy[:, 0] + x_offset))  # Adjust x coordinate
        box.append(float(detection.xyxy[:, 1] + y_offset))  # Adjust y coordinate
        box.append(float(detection.xywh[:, 2]))  # Adjust x coordinate
        box.append(float(detection.xywh[:, 3]))  # Adjust y coordinate
        box.append(float(detection.conf))
        detections_adjusted.append(box)
    return detections_adjusted

def process_image(raw_img, conf, save_path, class_info, plot = False):
    """
    Processes an image by splitting it into sub-regions, running object detection, and adjusting coordinates.

    Parameters:
    - raw_img (array): Original image to process.
    - conf (float): Confidence threshold for the detection model.
    - save_path (str): Path to save the output image with detections, if plotting is enabled.
    - class_info (dict): Dictionary containing class labels and other metadata.
    - plot (bool): If True, the function plots and saves the image with detection boxes.

    Returns:
    - results (list): List of adjusted detections for all image sub-regions.
    """
    processed_img = preprocess_image(raw_img)
    # if using SDC might want to enable this
    split = False
    if split:
        split_images = split_image(processed_img)
    else:
        split_images = [(processed_img,0,0)]

    results = []
    

    for img, x_offset, y_offset in split_images:

        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

        #Suppress output to prevent matlab crashing slidebook
        #with open(os.devnull, 'w') as nullfile:
        #     with contextlib.redirect_stdout(nullfile):

        results_split = model(img, conf = conf, device = "cpu")

        # Adjust coordinates
        if len(results_split[0].boxes.xyxy)!= 0:
            results_adjusted = adjust_coordinates(results_split[0].boxes, x_offset, y_offset)
            results.extend(results_adjusted)
        
    if plot:
        class_names = {key: value[0] for key, value in class_info.items()}
        plot_image_with_results(processed_img, results, class_names, class_info, save_path)

    print("return processs image")
    return results

def process_image_from_path(image_path, conf, save_path, plot = False):
    """
    Reads an image from a file path and processes it to obtain adjusted detections.

    Parameters:
    - image_path (str): Path to the image file.
    - conf (float): Confidence threshold for the detection model.
    - save_path (str): Path to save the output image with detections, if plotting is enabled.
    - class_info (dict): Dictionary containing class labels and other metadata.
    - plot (bool): If True, the function plots and saves the image with detection boxes.

    Returns:
    - results (list): List of adjusted detections for the processed image.
    """
    img = cv2.imread(image_path)
    results = process_image(img, conf, save_path, plot)

    return results



def process_single_location(x,y,z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info):
    """
    Processes a single image location to extract and convert detections into physical coordinates.

    Parameters:
    - x, y, z: Coordinates of the image location.
    - image: 2D array of the grayscale image to be processed.
    - xy_pixel_spacing, z_spacing: Pixel spacing values for xy-plane and z-dimension.
    - x_stage_direction, y_stage_direction, z_stage_direction: Stage direction multipliers.
    - LLSM: Boolean flag for adjustments specific to Lattice Light Sheet Microscopy, affects what z is used as doesn't need to be changed as well as stage direction.
    - z_offset: Offset for z-coordinate adjustments.
    - class_info: Dictionary with class configuration for filtering and ranking.

    Returns:
    - List of converted detections with physical coordinates, confidence scores, and class information.
    """
    new_z = z+z_offset
    if len(image.shape)==3:
        image = np.max(image, z)

    height, width = image.shape

    img_path = get_outimg_path()

    class_names = {key: value[0] for key, value in class_info.items()}

    results = process_image(image, 0.01, img_path, class_info , plot = False)



    # Filter and sort the detections
    filtered_sorted_detections = filter_and_sort_detections(results, class_info)

    converted_results = []
    for result in filtered_sorted_detections:
        class_id,im_x,im_y,w,h,conf = result
        class_name = class_names[class_id]
        converted_results.append(image_cordinates_to_physical(x, y, im_x, im_y, width, height, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name))

    print("return process single location")
    return converted_results

def image_cordinates_to_physical(x, y, im_x, im_y, w, h, new_z, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, class_id, conf, class_name):

    """
    Converts detection coordinates from image pixel space to real-world physical coordinates.

    Parameters:
    - x, y: Centered physical coordinates of the image.
    - im_x, im_y: Coordinates of the detected object within the image in pixels.
    - w, h: Width and height of the bounding box around the detected object in pixels.
    - new_z: Adjusted z-coordinate for the detected object in physical space.
    - xy_pixel_spacing: Physical distance represented by one pixel in the xy-plane.
    - z_spacing: Physical spacing in the z-dimension.
    - x_stage_direction, y_stage_direction, z_stage_direction: Direction multipliers (1 or -1) for the stage's orientation in each dimension.
    - LLSM: Boolean indicating if the adjustments are for Lattice Light Sheet Microscopy (LLSM), which uses a reversed y-axis.
    - class_id: Identifier for the detected object's class.
    - conf: Confidence score of the detection.
    - class_name: Name of the detected class.

    Returns:
    - A list containing [adjusted_x, adjusted_y, new_z, conf, class_id, class_name], where adjusted_x and adjusted_y are the physical coordinates of the detected object.
    """
    x_offset = im_x - w/2;
    y_offset = im_y - h/2;

    if LLSM:
        adjusted_x = x + x_offset * xy_pixel_spacing * x_stage_direction;
        adjusted_y = y + y_offset * xy_pixel_spacing * y_stage_direction*(-1);
    else:
        adjusted_x = x + x_offset * xy_pixel_spacing * x_stage_direction;
        adjusted_y = y + y_offset * xy_pixel_spacing * y_stage_direction;



    return [adjusted_x, adjusted_y, new_z, conf, class_id, class_name]

def merge_close_coordinates(coordinates, tolerance):
    """
    Merge coordinates that are within a specified tolerance and retain the class name from the first instance.

    Parameters:
    coordinates (array): Array of coordinates and class names.
    tolerance (float): The radius within which coordinates should be considered the same.

    Returns:
    tuple: Four separate lists for x, y, z coordinates, and class names of the unique locations.
    """
    unique_coords = []

    for coord in coordinates:
        x, y, z, conf, class_id, class_name = coord
        if not any(np.sqrt((x - uc[0])**2 + (y - uc[1])**2) <= tolerance for uc in unique_coords):
            unique_coords.append(coord)

    unique_coords = np.array(unique_coords, dtype=object)
    return unique_coords[:, 0], unique_coords[:, 1], unique_coords[:, 2], list(unique_coords[:, 5])


def filter_by_iou(detections):
    """
    Filters detections by removing overlaps with IoU > 0.5 or if one is completely inside another,
    keeping only the detection with the highest confidence.

    Parameters:
    detections (list): List of detections where each detection is a tuple (x, y, z, conf, class_id, class_name).

    Returns:
    list: Filtered detections.
    """
    def calculate_iou(det1, det2):
        """
        Calculate the Intersection over Union (IoU) for two detections.
        Each detection is represented by (x, y, z, width, height, depth).
        """
        x1, y1, z1, _, _, _ = det1
        x2, y2, z2, _, _, _ = det2

        # For simplicity, assuming a fixed radius for bounding spheres around (x, y, z)
        radius = 10  # Replace with an appropriate value or radius from your model
        dist_sq = (x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2
        return dist_sq < (2 * radius)**2

    filtered = []
    for i, det in enumerate(detections):
        keep = True
        for j, other_det in enumerate(detections):
            if i != j and calculate_iou(det, other_det):
                if det[3] < other_det[3]:  # Compare confidence scores
                    keep = False
                    break
        if keep:
            filtered.append(det)
    return filtered

def process_montage(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset):
    """
    Process a montage of images to extract and merge close physical coordinates and their class names.

    Parameters:
    X, Y, Z (list of floats): Lists of x, y, z coordinates.
    image (array): The complete montage image data.
    xy_pixel_spacing, z_spacing (float): Pixel dimensions in physical units.
    x_stage_direction, y_stage_direction, z_stage_direction (int): Stage direction multipliers.
    LLSM (bool): Flag for Lattice Light Sheet Microscopy specific adjustments.
    z_offset (float): Offset to apply to z coordinates.
    micron_tolerance (float): Distance tolerance in microns for merging coordinates.
    class_names (list of str): List of class names corresponding to each coordinate.

    Returns:
    tuple: Four arrays containing the new x, y, z coordinates, and their respective class names.
    """
    print("processing montage")
    # Based on imaging mode set different confidence thresholds for the model
    # Dict is classid: (classname, conf_threshold, ranking)
    # Setting a condfidence to 1.0 excludes it from results
    # Ranking determines order that results appear in MPL
    if LLSM:
        class_info = {
        0: ('prophase', 0.1, 0),
        1: ('earlyprometaphase', 0.2, 1),
        2: ('prometaphase', 1.0, 2),
        3: ('metapase', 1.0, 3),
        4: ('anaphse', 1.0, 4),
        5: ('telophase', 1.0, 5)
        }
    else:
        class_info = {
        0: ('prophase', 0.5, 0),
        1: ('earlyprometaphase', 0.5, 1),
        2: ('prometaphase', 0.5, 2),
        3: ('metapase', 0.5, 3),
        4: ('anaphse', 0.5, 4),
        5: ('telophase', 0.5, 5)
        }

    # Sometimes slidebook does not take montage correctly and only 1 image is
    # supplied so we check if there is just 1 value and then put them in lists
    results = []
    image_array = np.array(image)
    if type(X) == float:
        X = [X]
        Y = [Y]
        Z = [Z]
        tmp_image_array = image_array
        image_array = np.zeros(tmp_image_array.shape+(1,))
        image_array[:,:, 0] = tmp_image_array
    
    # start_time = time.time()
    # Process each image in montage
    for i, (x, y, z) in enumerate(zip(X, Y, Z)):
        tmp = process_single_location(x, y, z, image_array[:,:,i], xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset, class_info)
        if tmp:
            for item in tmp:
                results.append(item)
    
    # print(time.time()-start_time)
    
    if len(results) == 0:
        return np.array([]), np.array([]), np.array([]), []  # no results to process

    sorted_results = global_filter_and_sort_detections(results, class_info)

    filtered_results = filter_by_iou(sorted_results)

    

    final_result = np.array(filtered_results, dtype=object)

    # Remove very close coords
    new_X, new_Y, new_Z, new_class_names = merge_close_coordinates(final_result, 60)
    return new_X, new_Y, new_Z, new_class_names



def get_target_location(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, montage, z_offset):
    global experiment_path
    global today_date
    global repo_path
    global model
    global logging_directory
    today_date = datetime.now().strftime("%Y-%m-%d")
    repo = "C:\\Users\\Brook\\Documents\\Work\\CelFDrive-deployment-3i"
    	

    # Must be changed based on system, could have this come from MATLAB
    repo_path = repo
    print("Initialising model weights")
    model = YOLO(repo_path+"\\models\\SiRDNA\\\yolov11l_ALLDATA_UNDERSAMPLE\\weights\\best.pt") # absolute path
    print("Model weights successfully initialised")
    logging_directory = repo_path+f"\\Logging\\{today_date}" # absolute path

    # lattice model path model = YOLO("C:\\Users\\McAinsh\\Documents\\MATLAB\\yolov8\\Current\\best.pt")

    # Example function that prints the arguments
    """
    returns
    - list of x
    - list of y
    - list of z
    - list of capture scripts: if nothing is found 'find-loc-newloc', if something is found'find-loc-out'
    - list of names for next capture: Class found {class} or Nothing found
    - list of comments
    """
    verbose = True
    if verbose:
        print(f"x: {X}")
        print(f"y: {Y}")
        print(f"z: {Z}")
        print(f"image: {image}")
        print(f"xy_pixel_spacing: {xy_pixel_spacing}")
        print(f"z_spacing: {z_spacing}")
        print(f"x_stage_direction: {x_stage_direction}")
        print(f"y_stage_direction: {y_stage_direction}")
        print(f"z_stage_direction: {z_stage_direction}")
        print(f"LLSM: {LLSM}")
        print(f"montage: {montage}")

    # Ensure the logging directory exists (simulate the given directory structure)
    os.makedirs(logging_directory, exist_ok=True)
    # Create the next experiment folder and get its path
    create_exp_folder(logging_directory)


    if montage:
        new_X,new_Y,new_Z, class_list = process_montage(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset)
    else:
        new_X,new_Y,new_Z = process_single_location(X,Y,Z, image, xy_pixel_spacing, z_spacing, x_stage_direction, y_stage_direction, z_stage_direction, LLSM, z_offset)

    # example data
    # return 2, [X, X+1000], [Y, Y+1000], [Z, Z+1000], ["scriptname1", "scriptname2"], ["name1", "name2"], ["comments1", "comments2"]
    N = len(new_X)

    # if there are no results set the location to be the current location and set the script to be one with low led os it essentially does nothing
    if N == 0:
        N = 1
        script_list = ["donothing"]
        new_X = np.array([X])
        new_Y = np.array([Y])
        new_Z = np.array([Z])
        name_list = [f"No locations found."]
        comment_list = ["No locations found."]
        if montage:
            new_X = np.array([X[0]])
            new_Y = np.array([Y[0]])
            new_Z = np.array([Z[0]])

    else:
        #
        if LLSM:
            script_list = ["find-loc-out" for i in range(N)]
        else:
            script_list = ["floifmHighres" for i in range(N)]
        name_list = [f"{class_list[i]} Highres x {new_X[i]}, Y {new_Y[i]}, Z {new_Z[i]} " for i in range(N)]
        comment_list = ["Highres" for i in range(N)]

    
    return N, new_X, new_Y, new_Z, script_list, name_list, comment_list
