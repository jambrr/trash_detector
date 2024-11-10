import torch
from torchvision.ops import box_iou
from ultralytics import YOLO

def evaluate_yolo(model, dataloader, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate YOLO model on test data.

    Parameters:
    - model: YOLO model (PyTorch format, either YOLOv5, YOLOv7, or similar)
    - dataloader: DataLoader with test data
    - iou_threshold: IoU threshold to determine true positives
    - conf_threshold: Confidence threshold for detections

    Returns:
    - mAP, precision, recall: Evaluation metrics
    """

    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Initialize counters for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_ious = []
    
    # Loop through test images
    for images, targets in dataloader:
        images = images.to(device)
        targets = [t.to(device) for t in targets]
        
        # Perform inference
        with torch.no_grad():
            predictions = model(images)
        
        for i, pred in enumerate(predictions):
            # Filter predictions based on confidence threshold
            pred = pred[pred[:, 4] > conf_threshold]
            pred_boxes = pred[:, :4]  # Extract predicted bounding boxes
            pred_scores = pred[:, 4]  # Extract confidence scores
            pred_labels = pred[:, 5]  # Extract predicted labels
            
            # Get true boxes and labels for current image
            true_boxes = targets[i][:, :4]
            true_labels = targets[i][:, 4]
            
            # Calculate IoUs
            ious = box_iou(pred_boxes, true_boxes)
            max_iou, max_iou_idx = ious.max(dim=1)
            
            # Determine true positives and false positives
            for j, iou in enumerate(max_iou):
                if iou > iou_threshold and pred_labels[j] == true_labels[max_iou_idx[j]]:
                    true_positives += 1
                else:
                    false_positives += 1
            
            # Count false negatives
            false_negatives += len(true_boxes) - true_positives

    # Calculate precision, recall, and mAP
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    mAP = true_positives / len(dataloader.dataset)  # Simplified mAP approximation

    return mAP, precision, recall

def evaluate_yolo_without_dataloader(model, images, annotations, iou_threshold=0.5, conf_threshold=0.5):
    """
    Evaluate YOLO model on test data without a DataLoader.

    Parameters:
    - model: YOLO model (in PyTorch format)
    - images: List of images (each image is a torch.Tensor)
    - annotations: List of ground truth annotations for each image
                   (each annotation is a torch.Tensor containing bounding boxes and labels)
    - iou_threshold: IoU threshold to determine true positives
    - conf_threshold: Confidence threshold for detections

    Returns:
    - mAP, precision, recall: Evaluation metrics
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Initialize counters for true positives, false positives, and false negatives
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Loop through test images
    for img, target in zip(images, annotations):
        img = img.to(device).unsqueeze(0)  # Add batch dimension
        target = target.to(device)

        # Perform inference
        with torch.no_grad():
            pred = model(img)[0]  # Assume model output is a single batch of predictions

        # Filter predictions based on confidence threshold
        pred = pred[pred[:, 4] > conf_threshold]
        pred_boxes = pred[:, :4]  # Extract predicted bounding boxes
        pred_scores = pred[:, 4]  # Extract confidence scores
        pred_labels = pred[:, 5]  # Extract predicted labels

        # Get true boxes and labels for current image
        true_boxes = target[:, :4]
        true_labels = target[:, 4]

        # Calculate IoUs
        ious = box_iou(pred_boxes, true_boxes)
        max_iou, max_iou_idx = ious.max(dim=1)

        # Determine true positives and false positives
        for j, iou in enumerate(max_iou):
            if iou > iou_threshold and pred_labels[j] == true_labels[max_iou_idx[j]]:
                true_positives += 1
            else:
                false_positives += 1

        # Count false negatives
        false_negatives += len(true_boxes) - true_positives

    # Calculate precision, recall, and mAP
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    mAP = true_positives / len(images)  # Simplified mAP approximation

    return mAP, precision, recall

def test_image(model, image_path):
    model = YOLO(model)
    
    results = model.predict(image_path)
    
    from PIL import Image
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image
    
        # Show results to screen (in supported environments)
        r.show()
    
        # Save results to disk
        r.save(filename=f"results{i}.jpg")


