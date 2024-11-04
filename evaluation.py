import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import polygon

def test_class_performance(model, dataset_json_path, images_folder, class_id, iou_threshold=0.5):
    """
    Tests the model performance on a specific class using segmentation masks, visualizing images where the model fails.
    
    Parameters:
    - model: The segmentation model to test.
    - dataset_json_path (str): Path to the COCO-style JSON file with annotations.
    - images_folder (str): Folder containing the images referenced in the JSON file.
    - class_id (int): The class ID to test and visualize.
    - iou_threshold (float): The IoU threshold to consider a detection as correct.
    """
    # Load annotations
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
    
    # Filter out images and annotations for the specific class
    class_annotations = [ann for ann in data['annotations'] if ann['category_id'] == class_id]
    image_ids_with_class = {ann['image_id'] for ann in class_annotations}
    images_with_class = [img for img in data['images'] if img['id'] in image_ids_with_class]

    # Loop through each image with the target class
    for image_info in images_with_class:
        image_path = f"{images_folder}/{image_info['file_name']}"
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Run the model on the image (assuming model outputs a segmentation mask for the class)
        predicted_mask = model.predict_segmentation(image, class_id)  # Adjust this to your model’s method

        # Get ground truth masks for the specified class
        gt_masks = [
            segmentation_to_mask(ann['segmentation'], image.shape[:2])
            for ann in class_annotations if ann['image_id'] == image_info['id']
        ]
        
        # Calculate IoU between the predicted mask and each ground truth mask
        max_iou = max(calculate_mask_iou(predicted_mask, gt_mask) for gt_mask in gt_masks)
        
        # Visualization for poor detections
        if max_iou < iou_threshold:
            plot_segmentation(image, gt_masks, predicted_mask, max_iou)

def segmentation_to_mask(segmentation, image_shape):
    """
    Converts COCO-style polygon segmentation to a binary mask.
    
    Parameters:
    - segmentation (list): A list of lists containing polygon coordinates.
    - image_shape (tuple): Shape of the image (height, width).
    
    Returns:
    - np.array: Binary mask of the segmentation.
    """
    mask = np.zeros(image_shape, dtype=np.uint8)
    for poly in segmentation:
        poly = np.array(poly).reshape(-1, 2)
        rr, cc = polygon(poly[:, 1], poly[:, 0], image_shape)
        mask[rr, cc] = 1
    return mask

def calculate_mask_iou(mask1, mask2):
    """
    Calculates the IoU between two binary masks.
    
    Parameters:
    - mask1, mask2 (np.array): Binary masks to compare.
    
    Returns:
    - float: IoU between the two masks.
    """
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union != 0 else 0

def plot_segmentation(image, gt_masks, predicted_mask, iou):
    """
    Visualizes the image with ground truth and predicted segmentation masks for analysis.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot ground truth masks
    gt_overlay = image.copy()
    for gt_mask in gt_masks:
        gt_overlay[gt_mask > 0] = [0, 255, 0]  # Green overlay for GT
    plt.imshow(cv2.addWeighted(image, 0.5, gt_overlay, 0.5, 0))
    
    # Plot predicted mask
    pred_overlay = image.copy()
    pred_overlay[predicted_mask > 0] = [255, 0, 0]  # Red overlay for prediction
    plt.imshow(cv2.addWeighted(image, 0.5, pred_overlay, 0.5, 0))
    
    plt.title(f"IoU: {iou:.2f} (green=GT, red=Prediction)")
    plt.show()

