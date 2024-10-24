import json
import os
import re
import shutil
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split

#Data Preprocessing
def convert_2_yolo(json_file, output_dir="./datasets/TACO/"):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        file_name = str.lower(image_info['file_name'])
        #formatted_filename = re.search(r"(\d+)\.jpg", file_name).group(1)
        formatted_filename = os.path.splitext(filename_formatting(file_name))[0] + '.txt'
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

# Example usage

if __name__== "__main__":
    batch_annotations = "./datasets/TACO/batch_1/annotations.json"
    all_annotations = "./datasets/TACO/annotations.json"
    output_dir = "./datasets/TACO/all_labels"

    empty_directory("./datasets/TACO/val/images")
    empty_directory("./datasets/TACO/val/labels")
    empty_directory("./datasets/TACO/train/images")
    empty_directory("./datasets/TACO/train/labels")

    #convert_2_yolo(all_annotations, output_dir)
    #move_and_rename(all_annotations)
    #move_and_rename(all_annotations)
    #data_splitting()
