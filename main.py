from ultralytics import YOLO
from ultralytics.data.converter import convert_coco
import json
import os
import re
from pycocotools.coco import COCO

def get_classes():
    data_source = COCO(annotation_file='./datasets/TACO/annotations.json')
    
    img_ids = data_source.getImgIds()
    
    catIds = data_source.getCatIds()
    categories = data_source.loadCats(catIds)
    categories.sort(key=lambda x: x['id'])
    classes = {}
    coco_labels = {}
    coco_labels_inverse = {}
    for c in categories:
        coco_labels[len(classes)] = c['id']
        coco_labels_inverse[c['id']] = len(classes)
        classes[c['name']] = len(classes)
    
    class_num = {}
    print(classes)

if __name__ == "__main__":
    model = YOLO("yolo11m-seg.pt")
    model.train(
        data="./new_data.yaml",
        epochs=100,
        patience=10,
        imgsz=416,
        batch=-1,
        optimizer="SGD",
        lr0=0.001,
        cos_lr=True,
        auto_augment=None
    )
