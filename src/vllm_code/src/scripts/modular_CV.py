import cv2
import torch
from detectron2.config import get_cfg
from detic import add_detic_config
from detic.checkpoint import DetectionCheckpointer
from detic.predictor import DefaultPredictor

def setup_detic():
    """
    Setup Detic configuration for one-shot object detection.
    """
    cfg = get_cfg()
    add_detic_config(cfg)  # Add Detic-specific configs
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.merge_from_file("./configs/Detic/Base-R50.yaml")  # Path to Detic config file
    cfg.MODEL.WEIGHTS = "https://dl.fbaipublicfiles.com/detic/Base-R50.pkl"  # Pretrained weights
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection confidence threshold
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "datasets/metadata/conceptnet_embeddings.pth"
    cfg.MODEL.ROI_BOX_HEAD.CUSTOM_PROMPT = False
    return cfg


def detect_objects_in_camera_feed_detic(target_object, frame_limit=15):
    """
    Detects objects in a live camera feed, draws bounding boxes, calculates centroids,
    and returns centroids for the target object using Detic.

    Args:
        target_object (str): The name of the object to detect (e.g., "cardboard box").
        frame_limit (int): Number of frames to process (default: 15).
    
    Returns:
        List[Tuple[int, int]]: List of centroids (x, y) for detected objects across frames.
    """
    # Setup Detic model
    cfg = setup_detic()
    predictor = DefaultPredictor(cfg)

    # Open the camera feed
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to access the camera.")

    # Set the camera resolution to 720x720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 720)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    centroids_all_frames = []
    frame_count = 0

    try:
        while frame_count < frame_limit:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Exiting.")
                break

            # Perform object detection using Detic
            outputs = predictor(frame)
            instances = outputs["instances"].to("cpu")
            scores = instances.scores.numpy()
            classes = instances.pred_classes.numpy()
            boxes = instances.pred_boxes.tensor.numpy()

            centroids = []

            # Iterate through detections and draw bounding boxes
            for i, box in enumerate(boxes):
                class_id = classes[i]
                label = predictor.metadata.get("thing_classes")[class_id]  # Get class name

                if label == target_object:
                    x_min, y_min, x_max, y_max = map(int, box)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Calculate centroid
                    centroid_x = (x_min + x_max) // 2
                    centroid_y = (y_min + y_max) // 2
                    centroids.append((centroid_x, centroid_y))

                    # Annotate the frame
                    cv2.circle(frame, (centroid_x, centroid_y), 5, (0, 0, 255), -1)
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Store centroids for this frame
            centroids_all_frames.extend(centroids)

            # Display the frame
            cv2.imshow('Live Object Detection (Detic - 720p Camera)', frame)

            # Break if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

    finally:
        # Release the camera and close windows
        cap.release()
        cv2.destroyAllWindows()
    
    return centroids_all_frames


# Example usage
target_object = 'cardboard box'  # Replace with the desired object name
frame_limit = 10  # Number of frames to process
centroids = detect_objects_in_camera_feed_detic(target_object, frame_limit)
print("Centroids of detected objects across frames:", centroids)
