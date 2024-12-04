#!/usr/bin/env python3
import cv2
import numpy as np
import random
from collections import Counter
import rospy
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest, SetJointPosition, SetJointPositionRequest
from geometry_msgs.msg import Pose, Point, Quaternion
import tf.transformations as tf_tr
import math

def euler_to_quaternion(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    return Quaternion(*tf_tr.quaternion_from_euler(roll, pitch, yaw))

def control_gripper(position):
    rospy.wait_for_service('/goal_tool_control')
    try:
        tool_control = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        request.joint_position.position = [position]
        tool_control(request)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def compute_orientation(x, y, z):
    if z > 0.15:
        return (0, 0, 0)
    return (0, 80, 0)

def send_pose(poses):
    rospy.init_node('pose_gripper_control_node', anonymous=True)
    rospy.wait_for_service('/goal_task_space_path')
    set_pose = rospy.ServiceProxy('/goal_task_space_path', SetKinematicsPose)

    for position, gripper_state, delay in poses:
        x, y, z = position
        orientation = compute_orientation(x, y, z)
        pose = Pose(position=Point(x, y, z), orientation=euler_to_quaternion(*orientation))
        
        try:
            kinematics_pose = SetKinematicsPoseRequest()
            kinematics_pose.kinematics_pose.pose = pose
            kinematics_pose.planning_group = "arm"
            kinematics_pose.end_effector_name = "gripper"
            kinematics_pose.path_time = 2
            set_pose(kinematics_pose)
            rospy.sleep(delay)
            if gripper_state != "unchanged":
                control_gripper(0.01 if gripper_state == "open" else -0.01)
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        rospy.sleep(2)

def generate_poses(from_position, to_position):
    """
    Generate a sequence of poses to move from one position to another.
    """
    x_from, y_from = from_position
    x_to, y_to = to_position
    return [
        ((0.08, 0.0, 0.25), "open", 0.2),   # Move to initial hover position
        ((x_from, y_from, 0.15), "open", 0.2),  # Move above the 'from' position
        ((x_from, y_from, 0.05), "open", 0.2),  # Move to the 'from' position
        ((x_from, y_from, 0.05), "close", 0.2), # Close gripper at 'from'
        ((x_from, y_from, 0.12), "close", 0.2), # Lift from the 'from' position
        ((x_to, y_to, 0.15), "close", 0.2),     # Move above the 'to' position
        ((x_to, y_to, 0.08), "close", 0.2),     # Move to the 'to' position
        ((x_to, y_to, 0.08), "open", 0.2),      # Open gripper at 'to'
        ((x_to, y_to, 0.15), "open", 0.2),      # Lift from the 'to' position
        ((0.08, 0.0, 0.25), "open", 0.2),   # Return to hover position
    ]

# Global dictionary to store detected shapes
shapes_dict = {}

# Function to detect rectangles
def detect_rectangles(frame, original_frame, offset_x, offset_y):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)    # Apply Gaussian Blur
    edges = cv2.Canny(blurred, 70, 200)            # Detect edges using Canny

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.045 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # If the polygon has 4 sides, it is a rectangle
            x, y, w, h = cv2.boundingRect(approx)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + offset_x
                cy = int(M["m01"] / M["m00"]) + offset_y

                # Classify the color based on 500 random points
                color_name = get_most_common_color(frame, contour)

                # Add to global dictionary
                add_shape_to_dict("Rectangle", color_name, [cx, cy])

                # Draw the centroid and annotate it
                cv2.circle(original_frame, (cx, cy), 5, (255, 255, 255), -1)
                cv2.putText(original_frame, f"{color_name}", (cx + 10, cy - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(original_frame, f"({cx},{cy})", (cx + 10, cy + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.drawContours(original_frame, [approx + [offset_x, offset_y]], -1, (0, 255, 0), 3)

    return frame

# Function to detect circles
def detect_circles(frame, original_frame, offset_x, offset_y):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.medianBlur(gray, 5)              # Apply median blur

    # Use HoughCircles to detect circles
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1, 
        minDist=30, 
        param1=100, 
        param2=40, 
        minRadius=10, 
        maxRadius=300
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            x, y, radius = circle
            cx, cy = x + offset_x, y + offset_y

            # Create a circular mask for the circle
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), radius, 255, -1)

            # Classify the color based on 500 random points
            color_name = get_most_common_color(frame, mask)

            # Add to global dictionary
            add_shape_to_dict("Circle", color_name, [cx, cy])

            # Draw the centroid, circle, and annotate it
            cv2.circle(original_frame, (cx, cy), 5, (255, 255, 255), -1)
            cv2.putText(original_frame, f"{color_name}", (cx + 10, cy - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(original_frame, f"({cx},{cy})", (cx + 10, cy + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.circle(original_frame, (cx, cy), radius, (255, 0, 0), 3)

    return frame

# Function to classify color based on the dominant channel
def classify_color(color):
    max_channel = np.argmax(color)
    if max_channel == 0:
        return "Blue"
    elif max_channel == 1:
        return "Green"
    else:
        return "Red"

# Function to get the most common color from 500 random points
def get_most_common_color(frame, contour_or_mask):
    if isinstance(contour_or_mask, np.ndarray) and contour_or_mask.ndim == 2:  # Mask
        mask = contour_or_mask
        points = np.column_stack(np.where(mask > 0))
    else:  # Contour
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour_or_mask], -1, 255, -1)
        points = np.column_stack(np.where(mask > 0))

    # Choose up to 500 random points from the mask
    if len(points) > 500:
        points = points[np.random.choice(len(points), 500, replace=False)]

    # Classify each point and count occurrences
    colors = [classify_color(frame[point[0], point[1]]) for point in points]
    most_common_color = Counter(colors).most_common(1)[0][0]
    return most_common_color

# Function to add shape to global dictionary
def add_shape_to_dict(shape_type, color_name, centroid):
    global shapes_dict
    # Check if a shape with the same centroid already exists
    for key, value in shapes_dict.items():
        if value[1] == centroid:
            return
    shape_id = f"{shape_type}{len(shapes_dict) + 1}"
    shapes_dict[shape_id] = [color_name, centroid]

# Access camera
cap = cv2.VideoCapture(1)

# Set resolution of the camera frame
frame_width = 720
frame_height = 720
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# Define the region of interest (ROI)
roi_x_start, roi_x_end = 174, 690
roi_y_start, roi_y_end = 0, 410

frame_count = 0

from scipy.stats import mode

from statistics import mean

def group_shapes_by_proximity(shapes_dict, proximity_threshold=60):
    grouped_shapes = {}
    processed_centroids = set()

    def is_within_proximity(c1, c2):
        return abs(c1[0] - c2[0]) <= proximity_threshold and abs(c1[1] - c2[1]) <= proximity_threshold

    # Iterate over shapes and group by proximity
    group_id = 1
    for shape1, data1 in shapes_dict.items():
        if tuple(data1[1]) in processed_centroids:
            continue  # Skip already grouped centroids

        group = [shape1]  # Start a new group with the current shape
        processed_centroids.add(tuple(data1[1]))

        # Compare with other shapes
        for shape2, data2 in shapes_dict.items():
            if tuple(data2[1]) not in processed_centroids:
                if is_within_proximity(data1[1], data2[1]):
                    group.append(shape2)
                    processed_centroids.add(tuple(data2[1]))

        # Collect shapes, colors, and centroids for the group
        group_shapes = [s.split('_')[0] for s in group]  # Extract shape type (e.g., "Rectangle")
        group_colors = [shapes_dict[s][0] for s in group]  # Colors
        group_centroids = [shapes_dict[s][1] for s in group]  # Centroids

        # Select the most common shape and color
        mode_shape = Counter(group_shapes).most_common(1)[0][0]
        mode_color = Counter(group_colors).most_common(1)[0][0]

        # Calculate the mean centroid
        avg_centroid = [
            int(round(mean(c[0] for c in group_centroids))),
            int(round(mean(c[1] for c in group_centroids))),
        ]

        # Use the most common shape as the group name
        grouped_shapes[f"{mode_shape.lower()}_{group_id}"] = [mode_color, avg_centroid]
        group_id += 1

    return grouped_shapes

grouped_shapes = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw a thin outline rectangle for the ROI
    cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 1)

    # Crop the region of interest (ROI)
    roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

    # Process only the ROI
    roi_with_rectangles = detect_rectangles(roi, frame, roi_x_start, roi_y_start)
    roi_with_circles = detect_circles(roi_with_rectangles, frame, roi_x_start, roi_y_start)

    # Show the frame
    cv2.imshow("Shapes (Rectangles and Circles)", frame)
    
    frame_count += 1
    # After the frame processing loop, group and display the shapes
    if frame_count == 200:
        grouped_shapes = group_shapes_by_proximity(shapes_dict)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

def transform_centroids_to_meters(grouped_shapes):
    """
    Transform all centroid coordinates in the grouped_shapes dictionary from pixels to meters.
    xm = ypx * 0.000556 + 3
    ym = (xpx - 443) * 0.000566
    """
    transformed_shapes = {}
    for key, value in grouped_shapes.items():
        color, centroid_px = value
        xpx, ypx = centroid_px

        # Apply transformation equations
        xm = ypx * 0.000556 + 0.055
        ym = (xpx - 440) * 0.000556

        # Store the transformed centroids in meters
        transformed_shapes[key] = [color, [xm, ym]]

    return transformed_shapes

# Transform centroids to meters
transformed_grouped_shapes = transform_centroids_to_meters(grouped_shapes)

# Print the transformed shapes with centroids in meters
print("Transformed Grouped Shapes (Centroids in Meters):")
print((transformed_grouped_shapes))

cap.release()
cv2.destroyAllWindows()


# from transformers import pipeline

# # Initialize the Hugging Face pipeline for zero-shot classification
# classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# # Function to process a command and find the shape
# def find_shape_by_command(command, grouped_shapes):
#     # Prepare candidate labels for zero-shot classification
#     candidate_labels = [f"{key.split('_')[0]} {value[0].lower()}" for key, value in grouped_shapes.items()]
    
#     # Use the classifier to analyze the command
#     result = classifier(command, candidate_labels)
    
#     # Get the most likely label
#     top_label = result["labels"][0]
#     top_score = result["scores"][0]

#     # Ensure the confidence score is sufficiently high (e.g., > 0.7)
#     if top_score > 0.9:
#         # Parse the shape and color from the top label
#         shape, color = top_label.split()
        
#         # Find the corresponding item in the grouped_shapes dictionary
#         for key, value in grouped_shapes.items():
#             if shape in key and value[0].lower() == color:
#                 print(f"Centroid of the {color.capitalize()} {shape.capitalize()}: {value[1]}")
#                 return value[1]
    
#     print("No matching shape found.")ed__shapes
#     return None

# # Main loop to process commands repetitively
# def main():
#     while True:
#         command = input("Enter your command (or type 'stop' to quit): ").strip()
#         if any(i in command.lower() for i in ["stop", "enough", "home"]):
#             print("Exiting the program. Goodbye!")
#             break
#         find_shape_by_command(command, transformed_grouped_shapes)

from transformers import pipeline

# Initialize the Hugging Face pipeline for zero-shot classification
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def process_place_command_simple(command, transformed_grouped_shapes):
    """
    Process a 'place' command and return the source and destination coordinates.
    The LLM runs twice: once for the source and once for the destination.
    """
    # Define the classifier
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Extract candidate labels for the shapes
    candidate_labels = [
        f"{key.split('_')[0]} {value[0].lower()}" for key, value in transformed_grouped_shapes.items()
    ]

    # Run the classifier for the "from" shape (first part of the command)
    from_result = classifier(command, candidate_labels)
    from_label = from_result["labels"][0]
    from_shape, from_color = from_label.split()

    # Run the classifier for the "to" shape (second part of the command)
    to_result = classifier(command, candidate_labels)
    to_label = to_result["labels"][1]
    to_shape, to_color = to_label.split()

    # Find the centroids in the transformed_grouped_shapes dictionary
    from_centroid = None
    to_centroid = None

    for key, value in transformed_grouped_shapes.items():
        shape = key.split('_')[0]  # Shape type (e.g., "rectangle", "circle")
        color = value[0].lower()  # Color (e.g., "red", "blue")

        # Match the "from" shape
        if shape == from_shape and color == from_color:
            from_centroid = value[1]

        # Match the "to" shape
        if shape == to_shape and color == to_color:
            to_centroid = value[1]

        # Break early if both centroids are found
        if from_centroid and to_centroid:
            break

    if from_centroid and to_centroid:
        print(f"Move from {from_centroid} (source) to {to_centroid} (destination).")
        return from_centroid, to_centroid

    # Handle cases where shapes are not found
    if not from_centroid:
        print(f"Could not find the 'from' shape: {from_shape} {from_color}")
    if not to_centroid:
        print(f"Could not find the 'to' shape: {to_shape} {to_color}")

    return None, None

def main():
    while True:
        command = input("Enter your command (or type 'stop' to quit): ").strip()
        if any(i in command.lower() for i in ["stop", "enough", "home"]):
            print("Exiting the program. Goodbye!")
            break
        
        if "place" in command.lower() and "on top of" in command.lower():
            from_coord, to_coord = process_place_command_simple(command, transformed_grouped_shapes)
            if from_coord and to_coord:
                poses = generate_poses(from_coord, to_coord)
                send_pose(poses)
        else:
            print("Command not recognized. Try again.")


# Run the program
if __name__ == "__main__":
    main()