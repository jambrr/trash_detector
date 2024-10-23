from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
import json
import os
import re

#Data Preprocessing
def convert_2_yolo(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        file_name = str.lower(image_info['file_name'])
        formatted_filename = re.search(r"(\d+)\.jpg", file_name).group(1)
        img_width = image_info['width']
        img_height = image_info['height']
        yolo_annotations = []

        class_id = ann['category_id']
        bbox = ann['bbox']  # Assuming bbox format is [xmin, ymin, xmax, ymax]
        x1, y1, x2, y2 = bbox
        
        # Calculate YOLO format values
        x_center = (x1 + x2) / 2.0 
        y_center = (y1 + y2) / 2.0
        width = (x2 - x1)
        height = (y2 - y1)

        x_center_normalized = (x_center - img_width / 2) / img_width
        y_center_normalized = (y_center - img_height / 2) / img_height
        width_normalized = width / img_width
        height_normalized = height / img_height

        # Prepare the YOLO format line
        yolo_annotations.append(f"{class_id} {x_center_normalized:.6f} {y_center_normalized:.6f} {width_normalized:.6f} {height_normalized:.6f}")

        # Write annotations to a file in the YOLO format
        output_file = os.path.join(output_dir, f"{formatted_filename}.txt")
        with open(output_file, 'w') as out_f:
            out_f.write("\n".join(yolo_annotations))
        print(f"YOLO format annotations saved for image {image_id}")

def analyse_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    classes = []

    for ann in data['annotations']:
        image_id = ann['image_id']
        image_info = data['images'][image_id]
        img_width = image_info['width']
        img_height = image_info['height']
        yolo_annotations = []

        ann['category_id']
        bbox = ann['bbox']  # Assuming bbox format is [xmin, ymin, xmax, ymax]

        print(f"YOLO format annotations saved for image {image_id}") 

if __name__ == "__main__":
    json_file = "./datasets/TACO/batch_1/annotations.json"
    output_dir = "./datasets/TACO/train/labels"

    #convert_2_yolo(json_file, output_dir)

    # Model Training
    model = YOLO("yolo11n.pt")
    model.train(data="./datasets/TACO/data.yaml", imgsz=800, epochs=20, batch=8)


