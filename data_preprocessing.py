import json
import os
import re
import shutil
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import cv2
from pycocotools import mask as maskUtils
import random

#Data Preprocessing
def convert_2_yolo(json_file, output_dir="./datasets/TACO/"):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    num = 0
    for ann in data['annotations']:
        num = num + 1
        print(num)
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        file_name = str.lower(image_info['file_name'])
        #formatted_filename = re.search(r"(\d+)\.jpg", file_name).group(1)
        #formatted_filename = os.path.splitext(filename_formatting(file_name))[0] + '.txt'
        formatted_filename = os.path.splitext(file_name)[0] + '.txt'
        img_width = image_info['width']
        img_height = image_info['height']
        yolo_annotations = []

        class_id = ann['category_id']
        segmentation = ann['segmentation'][0]
        bbox = ann['bbox']  # Assuming bbox format is [xmin, ymin, xmax, ymax]
        x1, y1, w, h = bbox
        
        # Normalize label values
        x_center = (x1 + w/2) / img_width
        y_center = (y1 + h/2) / img_height
        w = w / img_width
        h = h / img_height

        # Prepare the YOLO format line
        #yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        normalized_coords = normalize_segments(segmentation, img_width, img_height)
        row = f"{class_id} " + " ".join([f"{coord:.6f}" for coord in normalized_coords]) + "\n"
        yolo_annotations.append(row)

        # Write annotations to a file in the YOLO format
        output_file = os.path.join(output_dir, formatted_filename)
        with open(output_file, 'a') as out_f:
            out_f.write(row)

        print(f"YOLO format annotations saved for image {image_id}")

def filename_formatting(filename):
    basename, image_name = filename.split("/")
    new_filename = str.lower(f"{basename}-{image_name}")

    return new_filename

def move_and_rename(filename, input_dir="./datasets/TACO/", output_dir="./datasets/TACO/all_images/"):
    with open(filename, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        filename = image_info['file_name']

        new_filename = filename_formatting(filename)

        src = os.path.join(input_dir, filename)
        dest = os.path.join(output_dir, new_filename)

        if os.path.exists(output_dir):
            shutil.copy(src, dest)
            print(f"Image moved to: {output_dir}")
        else:
            print(f"Source image not found: {input_dir}")

def normalize_segments(coords, img_width, img_height):
    normalized = []
    for i in range(0, len(coords), 2):
        x = coords[i] / img_width
        y = coords[i + 1] / img_height
        normalized.append(x)
        normalized.append(y)
    return normalized

def normalize_bbox(coords, img_width, img_height):
    normalized = []
    for i in range(0, len(coords), 2):
        x = coords[i] / img_width
        y = coords[i + 1] / img_height
        normalized.append(x)
        normalized.append(y)
 
def data_splitting(filename="./datasets/TACO/annotations.json", all_labels="./datasets/TACO/all_labels/", all_images="./datasets/TACO/all_images/", train_dir="./datasets/TACO/train/", val_dir="./datasets/TACO/val/"):
    with open(filename, 'r') as f:
        data = json.load(f)

    filenames = []
    labels = []

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        filename = filename_formatting(image_info['file_name'])
        label = os.path.splitext(filename)[0] + '.txt'

        filenames.append(filename)
        labels.append(label)


    filename_ds = np.array(filenames)
    labels_ds = np.array(labels)

    X_train, X_val, y_train, y_val = train_test_split(filename_ds, labels_ds, test_size=0.2, shuffle=True)

    for i in X_train:
        shutil.copy(all_images+i, train_dir+"images/"+i)
    for i in y_train:
        shutil.copy(all_labels+i, train_dir+"labels/"+i)
    for i in X_val:
        shutil.copy(all_images+i, val_dir+"images/"+i)
    for i in y_val:
        shutil.copy(all_labels+i, val_dir+"labels/"+i)

def empty_directory(directory_path):
    # Check if the directory exists
    if os.path.exists(directory_path):
        # Iterate over all the files and directories within the given directory
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            
            # Check if it's a file or directory
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  # Remove the file
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Remove the directory
                print(f"Deleted directory: {file_path}")
    else:
        print(f"Directory {directory_path} does not exist!")

def convert_image_bw(input_dir, label_dir, output_dir=None):
    # Ensure the output directory exists
    if output_dir is None:
        output_dir = input_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith((".jpg", ".png")):
            # Full path to the image
            img_path = os.path.join(input_dir, filename)
    
            # Load the image
            image = cv2.imread(img_path)
    
            if image is not None:  # Check if the image is loaded successfully
                # Convert the image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                new_filename = add_prefix("bw", filename)
    
                # Create the output file path
                output_path = os.path.join(output_dir, new_filename)
    
                # Save the grayscale image
                cv2.imwrite(output_path, gray_image)

                #old_label = os.path.splitext(filename)[0] + '.txt'
                #new_label = f"bw-{old_label}"

                # Add a new label file too
                #shutil.copy(label_dir+old_label, label_dir+new_label)
                print(f'Converted {filename} to grayscale.')
            else:
                print(f'Failed to load {filename}')
    
    copy_files_with_new_names(label_dir, prefix='bw_')
    print("All images have been processed.")

def copy_files_with_new_names(input_dir, output_dir=None, prefix='', suffix=''):
    # If no output directory is provided, use the input directory
    if output_dir is None:
        output_dir = input_dir

    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.startswith("original_"):
            file_path = os.path.join(input_dir, filename)

            # Check if the current file is a regular file (not a directory)
            if os.path.isfile(file_path):
                # Split the filename and extension
                file_name, file_ext = os.path.splitext(filename)

                # Create the new filename with prefix or suffix
                new_filename = f"{prefix}{file_name}{suffix}{file_ext}"

                # Create the new file path
                new_file_path = os.path.join(output_dir, new_filename)

                # Copy the file to the new location with the new name
                shutil.copy(file_path, new_file_path)
                print(f"Copied {filename} to {new_filename}")

    print("All files have been copied and renamed.")

def flip_image(image, direction):
    """
    Augments the image by flipping it vertically and horizontally.

    Parameters:
    - image: The input image as a NumPy array.

    Returns:
    - original: The original image.
    - vertical_flip: The vertically flipped image.
    - horizontal_flip: The horizontally flipped image.
    - both_flips: The image flipped both vertically and horizontally.
    """
    
    # Ensure the input is a valid image
    if image is None:
        raise ValueError("Input image is None.")
    
    # Flip the image vertically
    flipped_image = cv2.flip(image, direction)  # 0 means flipping around the x-axis

    return flipped_image

def flip_augmentation(directory):
    """
    Applies augmentation to images with the 'original_' prefix in the specified directory.

    Parameters:
    - directory: Path to the directory containing images.
    """
    for filename in os.listdir(directory):
        if filename.startswith("original_") and filename.endswith((".jpg", ".png")):
            # Load the image
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            
            if image is not None:
                # Perform augmentation
                vertical_flip = flip_image(image, 0)
                horizontal_flip = flip_image(image, 1)
                
                # Save augmented images with modified names
                cv2.imwrite(os.path.join(directory, f"vertical_flip_{filename}"), vertical_flip)
                cv2.imwrite(os.path.join(directory, f"horizontal_flip_{filename}"), horizontal_flip)
                
                print(f'Augmented {filename} and saved flips.')
            else:
                print(f'Failed to load {filename}')

def add_prefix(prefix, filename):
    return f"{prefix}_{filename}"

#Augmentation functions
def add_noise(image, noise_ratio=0.001):
    noisy_image = image.copy()
    num_noisy_pixels = int(noise_ratio * image.shape[0] * image.shape[1])

    x_coords = np.random.randint(0, image.shape[1], num_noisy_pixels)
    y_coords = np.random.randint(0, image.shape[0], num_noisy_pixels)

    noisy_image[y_coords, x_coords] = np.random.randint(0, 256, (num_noisy_pixels, 3))

    return noisy_image

def apply_noise(image_dir, label_dir, strength):
    for filename in os.listdir(image_dir):
        if filename.startswith("original_"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            noisy_image = add_noise(image, noise_ratio=strength)
            new_filename = add_prefix("noise", filename)
            new_filepath = os.path.join(image_dir, new_filename)
            cv2.imwrite(new_filepath, noisy_image)

            print("Creating noisy image")

    copy_files_with_new_names(label_dir, prefix='noise_')

def add_saturation(image, saturation_change=None):
    if saturation_change is None:
        saturation_change = np.random.uniform(-0.25, 0.25)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] *= (1 + saturation_change)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    adjusted_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return adjusted_image

def add_exposure(image, exposure_percent=0):
    exposure_percent = np.clip(exposure_percent, -10, 10)
    exposure_factor = 1 + (exposure_percent / 100.0)
    exposure_adjusted_image = cv2.convertScaleAbs(image, alpha=exposure_factor)

    return exposure_adjusted_image

def apply_exposure(image_dir, label_dir):
    for filename in os.listdir(image_dir):
        if filename.startswith("original_"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            exposed_image = add_exposure(image)
            new_filename = add_prefix("exposed", filename)
            new_filepath = os.path.join(image_dir, new_filename)
            cv2.imwrite(new_filepath, exposed_image)

            print("Creating noisy image")

    copy_files_with_new_names(label_dir, prefix='exposed_')

def apply_saturation(image_dir, label_dir):
    for filename in os.listdir(image_dir):
        if filename.startswith("original_"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)

            saturated_image = add_saturation(image)
            new_filename = add_prefix("saturated", filename)
            new_filepath = os.path.join(image_dir, new_filename)
            cv2.imwrite(new_filepath, saturated_image)

            print("Creating noisy image")

    copy_files_with_new_names(label_dir, prefix='saturated_')


def apply_seg_mask(image_dir, label_dir, output_dir=None, background_image_path=None):
    if output_dir is None:
        output_dir = image_dir

    for filename in os.listdir(image_dir):
        if filename.startswith("original_"):
            image_path = os.path.join(image_dir, filename)
            image = cv2.imread(image_path)
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            height, width = image.shape[:2]
            label_path = os.path.join(label_dir, os.path.splitext(filename)[0] + '.txt')

            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data = line.strip().split()
                    class_id = int(data[0])
                    polygon = np.array([[float(x) * width, float(y) * height] for x, y in zip(data[1::2], data[2::2])], dtype=np.int32)
                    cv2.fillPoly(mask, [polygon], color=(class_id + 1))

            masked_image = cv2.bitwise_and(image, image, mask=mask)

            if background_image_path:
                background = cv2.imread(background_image_path)
                background = cv2.resize(background, (image.shape[1], image.shape[0]))
            
            else: 
                background = np.zeros_like(image)
                background[:] = (0, 255, 0)

            inverted_mask = cv2.bitwise_not(mask)
            background_part = cv2.bitwise_and(background, background, mask=inverted_mask)
            final_image = cv2.add(masked_image, background_part)
            new_filename = add_prefix("no_background", filename)

            new_filepath = os.path.join(output_dir, new_filename)

            cv2.imwrite(new_filepath, final_image)

    copy_files_with_new_names(label_dir, prefix='no_background-')
   

def replace_classes_in_json(input_json_path, output_json_path, class_mapping):
    """
    Replaces class annotations in a COCO-style JSON file using a specified class mapping and writes the result to a new JSON file.

    Parameters:
    - input_json_path (str): Path to the input JSON file.
    - output_json_path (str): Path to the output JSON file where updated annotations will be saved.
    - class_mapping (dict): Dictionary mapping old class IDs to new class IDs.
    """
    # Load the original JSON data
    with open(input_json_path, 'r') as f:
        data = json.load(f)
    
    # Replace class IDs in each annotation
    for annotation in data.get('annotations', []):
        # Update 'category_id' if it exists in class_mapping
        if annotation['category_id'] in class_mapping:
            annotation['category_id'] = class_mapping[annotation['category_id']]
    
    # Write the updated data to a new JSON file
    with open(output_json_path, 'w') as f:
        json.dump(data, f, indent=4)
    
    print(f"Updated annotations saved to {output_json_path}")


def rename_files_with_prefix(directory, prefix="original_"):
    """
    Renames every file in the specified directory by adding a prefix to the filename.

    Parameters:
    - directory: The directory containing files to rename.
    - prefix: The prefix to add to each filename (default is "original_").
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise ValueError("Directory does not exist.")

    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Create the full path for the current file
        old_path = os.path.join(directory, filename)

        # Skip directories
        if os.path.isfile(old_path):
            # Create the new filename with the prefix
            new_filename = prefix + filename
            new_path = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed {filename} to {new_filename}')

import pickle

def find_images_without_labels(image_dir, label_dir, image_extensions=['.jpg', '.jpeg', '.png']):
    """
    Lists images in `image_dir` that have no corresponding label file in `label_dir`.

    Parameters:
    - image_dir (str): Path to the directory containing image files.
    - label_dir (str): Path to the directory containing label files.
    - image_extensions (list): List of image file extensions to consider (e.g., ['.jpg', '.png']).

    Returns:
    - List of image file names without corresponding label files.
    """
    # List all images in the image directory
    image_files = [f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in image_extensions]

    # List all label files in the label directory
    label_files = [os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')]

    # Find images without corresponding labels
    images_without_labels = [img for img in image_files if os.path.splitext(img)[0] not in label_files]

    return images_without_labels

if __name__== "__main__":
    batch_annotations = "./datasets/TACO/batch_1/annotations.json"
    all_annotations = "./datasets/TACO/annotations.json"
    all_labels = "./datasets/TACO/all_labels"
    all_images = "./datasets/TACO/all_images"
    
    #trashnet - training
    trashnet_img_output = "./datasets/trashnet/train/labels"
    trashnet_annotations = "./datasets/trashnet/train/_annotations.coco.json"

    #trashnet - validation
    trashnet_val_output = "./datasets/trashnet/valid/labels"
    trashnet_val_annotations = "./datasets/trashnet/valid/_annotations.coco.json"

    taco_aug_label_train = "./datasets/taconet/train/labels" 
    taco_aug_label_val = "./datasets/taconet/val/labels" 

    #empty_directory("./datasets/TACO/val/images")
    #empty_directory("./datasets/TACO/val/labels")
    #empty_directory("./datasets/TACO/train/images")
    #empty_directory("./datasets/TACO/train/labels")

    #convert_2_yolo("./datasets/trashnet/train/new_ann.json", output_dir="./datasets/trashnet/train/labels")
    convert_2_yolo("./datasets/trashnet/valid/new_ann.json", output_dir="./datasets/trashnet/valid/labels")
    #move_and_rename(all_annotations)
    #data_splitting()

    #convert_image_bw("./datasets/taconet/train/images", taco_aug_label_train)
    #convert_image_bw("./datasets/taconet/val/images", taco_aug_label_val)

    #copy_files_with_new_names("./datasets/TACO_aug/train/labels", prefix='bw-')
    #copy_files_with_new_names("./datasets/TACO_aug/val/labels", prefix='bw-')

    #apply_seg_mask("./datasets/waste/train/images", label_dir="./datasets/waste/train/labels")
    #apply_seg_mask("./datasets/waste/val/images", label_dir="./datasets/waste/val/labels")
    #rename_files_with_prefix("./datasets/waste/train/labels")
    #rename_files_with_prefix("./datasets/waste/train/images")
    #flip_augmentation("./datasets/TACO_aug/train/images")
    #copy_files_with_new_names("./datasets/TACO_aug/train/labels", prefix='-')

    #rename_files_with_prefix("./datasets/waste/val/images")
    #rename_files_with_prefix("./datasets/waste/val/labels")
    #flip_augmentation("./datasets/TACO_aug/val/images")
    #rename_files_with_prefix("./datasets/taco4/train/images", prefix='taco-')
    #rename_files_with_prefix("./datasets/taco4/train/labels", prefix='taco-')
    #rename_files_with_prefix("./datasets/taco4/val/images", prefix='taco-')
    #rename_files_with_prefix("./datasets/taco4/val/labels", prefix='taco-')

    #apply_noise("./datasets/TACO_aug/train/images", taco_aug_label_train, 0.3)
    #apply_noise("./datasets/TACO_aug/val/images", taco_aug_label_val, 0.3)

    #apply_exposure("./datasets/TACO_aug/train/images", taco_aug_label_train)
    #apply_exposure("./datasets/TACO_aug/val/images", taco_aug_label_val)
    #apply_saturation("./datasets/TACO_aug/train/images", taco_aug_label_train)

    import splitfolders
    #splitfolders.ratio('./datasets/taconet/tmp/', output="./datasets/taconet", seed=4321, ratio=(.8, .2))
    
    class_mapping = {
        0: 1,  # Aluminium foil -> Metals
        2: 1,  # Aluminium blister pack -> Metals
        8: 1,  # Metal bottle cap -> Metals
        10: 1, # Food Can -> Metals
        11: 1, # Aerosol -> Metals
        12: 1, # Drink can -> Metals
        28: 1, # Metal lid -> Metals
        50: 1, # Pop tab -> Metals
        52: 1, # Scrap metal -> Metals
        4: 4,   # Other plastic bottle -> Plastic
        5: 4,   # Clear plastic bottle -> Plastic
        7: 4,   # Plastic bottle cap -> Plastic
        21: 4,  # Disposable plastic cup -> Plastic
        22: 4,  # Foam cup -> Plastic
        24: 4,  # Other plastic cup -> Plastic
        27: 4,  # Plastic lid -> Plastic
        29: 4,  # Other plastic -> Plastic
        35: 4,  # Plastified paper bag -> Plastic
        36: 4,  # Plastic film -> Plastic
        37: 4,  # Six pack rings -> Plastic
        38: 4,  # Garbage bag -> Plastic
        39: 4,  # Other plastic wrapper -> Plastic
        40: 4,  # Single-use carrier bag -> Plastic
        41: 4,  # Polypropylene bag -> Plastic
        42: 4,  # Crisp packet -> Plastic
        43: 4,  # Spread tub -> Plastic
        44: 4,  # Tupperware -> Plastic
        45: 4,  # Disposable food container -> Plastic
        46: 4,  # Foam food container -> Plastic
        47: 4,  # Other plastic container -> Plastic
        48: 4,  # Plastic gloves -> Plastic
        49: 4,  # Plastic utensils -> Plastic
        54: 4,  # Squeezable tube -> Plastic
        55: 4,  # Plastic straw -> Plastic
        57: 4,  # Styrofoam piece -> Plastic
        3: 2,   # Carded blister pack -> Cardboard
        13: 2,  # Toilet tube -> Cardboard
        14: 2,  # Other carton -> Cardboard
        15: 2,  # Egg carton -> Cardboard
        16: 2,  # Drink carton -> Cardboard
        17: 2,  # Corrugated carton -> Cardboard
        18: 2,  # Meal carton -> Cardboard
        19: 2,  # Pizza box -> Cardboard
        20: 2,  # Paper cup -> Cardboard
        30: 2,  # Magazine paper -> Cardboard
        31: 2,  # Tissues -> Cardboard
        32: 2,  # Wrapping paper -> Cardboard
        33: 2,  # Normal paper -> Cardboard
        34: 2,  # Paper bag -> Cardboard
        56: 2,
        6: 6,  # Glass bottle -> Glass
        9: 6,  # Broken glass -> Glass
        23: 6, # Glass cup -> Glass
        26: 6,
        51: 3,  # Rope & strings -> Textile
        53: 3,
        1: 0,
        25: 7,
        58: 8,
        59: 5
    }

    class_mapping2 = {
        0:8,
        5:4,
        3:2,
    }

    
    # Call the function
    #replace_classes_in_json("./datasets/trashnet/valid/_annotations.coco.json", "./datasets/trashnet/valid/new_ann.json", class_mapping2)

    images = find_images_without_labels("datasets/taconet/train/images", "datasets/taconet/train/labels")
    for image in images:
        print(image)

    print(len(images))





