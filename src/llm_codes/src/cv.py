import cv2
import numpy as np
import random
from collections import Counter

# Function to detect rectangles
def detect_rectangles(frame, original_frame, offset_x, offset_y):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)    # Apply Gaussian Blur
    edges = cv2.Canny(blurred, 50, 150)            # Detect edges using Canny

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Approximate the contour to a polygon
        epsilon = 0.0454 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # If the polygon has 4 sides, it is a rectangle
            x, y, w, h = cv2.boundingRect(approx)
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + offset_x
                cy = int(M["m01"] / M["m00"]) + offset_y

                # Classify the color based on 500 random points
                color_name = get_most_common_color(frame, contour)

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
    
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

# import cv2
# import numpy as np
# import random
# from collections import Counter

# # Global dictionary to store detected shapes
# shapes_dict = {}

# # Function to detect rectangles
# def detect_rectangles(frame, original_frame, offset_x, offset_y):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     blurred = cv2.GaussianBlur(gray, (7, 7), 0)    # Apply Gaussian Blur
#     edges = cv2.Canny(blurred, 50, 150)            # Detect edges using Canny

#     contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#     for contour in contours:
#         # Approximate the contour to a polygon
#         epsilon = 0.0454 * cv2.arcLength(contour, True)
#         approx = cv2.approxPolyDP(contour, epsilon, True)

#         if len(approx) == 4:  # If the polygon has 4 sides, it is a rectangle
#             x, y, w, h = cv2.boundingRect(approx)
#             M = cv2.moments(contour)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"]) + offset_x
#                 cy = int(M["m01"] / M["m00"]) + offset_y

#                 # Classify the color based on 500 random points
#                 color_name = get_most_common_color(frame, contour)

#                 # Add to global dictionary
#                 add_shape_to_dict("Rectangle", color_name, [cx, cy])

#                 # Draw the centroid and annotate it
#                 cv2.circle(original_frame, (cx, cy), 5, (255, 255, 255), -1)
#                 cv2.putText(original_frame, f"{color_name}", (cx + 10, cy - 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#                 cv2.putText(original_frame, f"({cx},{cy})", (cx + 10, cy + 10), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

#             cv2.drawContours(original_frame, [approx + [offset_x, offset_y]], -1, (0, 255, 0), 3)

#     return frame

# # Function to detect circles
# def detect_circles(frame, original_frame, offset_x, offset_y):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#     blurred = cv2.medianBlur(gray, 5)              # Apply median blur

#     # Use HoughCircles to detect circles
#     circles = cv2.HoughCircles(
#         blurred, 
#         cv2.HOUGH_GRADIENT, 
#         dp=1, 
#         minDist=30, 
#         param1=100, 
#         param2=40, 
#         minRadius=10, 
#         maxRadius=300
#     )

#     if circles is not None:
#         circles = np.uint16(np.around(circles))
#         for circle in circles[0, :]:
#             x, y, radius = circle
#             cx, cy = x + offset_x, y + offset_y

#             # Create a circular mask for the circle
#             mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#             cv2.circle(mask, (x, y), radius, 255, -1)

#             # Classify the color based on 500 random points
#             color_name = get_most_common_color(frame, mask)

#             # Add to global dictionary
#             add_shape_to_dict("Circle", color_name, [cx, cy])

#             # Draw the centroid, circle, and annotate it
#             cv2.circle(original_frame, (cx, cy), 5, (255, 255, 255), -1)
#             cv2.putText(original_frame, f"{color_name}", (cx + 10, cy - 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.putText(original_frame, f"({cx},{cy})", (cx + 10, cy + 10), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#             cv2.circle(original_frame, (cx, cy), radius, (255, 0, 0), 3)

#     return frame

# # Function to classify color based on the dominant channel
# def classify_color(color):
#     max_channel = np.argmax(color)
#     if max_channel == 0:
#         return "Blue"
#     elif max_channel == 1:
#         return "Green"
#     else:
#         return "Red"

# # Function to get the most common color from 500 random points
# def get_most_common_color(frame, contour_or_mask):
#     if isinstance(contour_or_mask, np.ndarray) and contour_or_mask.ndim == 2:  # Mask
#         mask = contour_or_mask
#         points = np.column_stack(np.where(mask > 0))
#     else:  # Contour
#         mask = np.zeros(frame.shape[:2], dtype=np.uint8)
#         cv2.drawContours(mask, [contour_or_mask], -1, 255, -1)
#         points = np.column_stack(np.where(mask > 0))

#     # Choose up to 500 random points from the mask
#     if len(points) > 500:
#         points = points[np.random.choice(len(points), 500, replace=False)]

#     # Classify each point and count occurrences
#     colors = [classify_color(frame[point[0], point[1]]) for point in points]
#     most_common_color = Counter(colors).most_common(1)[0][0]
#     return most_common_color

# # Function to add shape to global dictionary
# def add_shape_to_dict(shape_type, color_name, centroid):
#     global shapes_dict
#     # Check if a shape with the same centroid already exists
#     for key, value in shapes_dict.items():
#         if value[1] == centroid:
#             return
#     shape_id = f"{shape_type}{len(shapes_dict) + 1}"
#     shapes_dict[shape_id] = [color_name, centroid]

# # Access camera
# cap = cv2.VideoCapture(1)

# # Set resolution of the camera frame
# frame_width = 720
# frame_height = 720
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# # Define the region of interest (ROI)
# roi_x_start, roi_x_end = 174, 690
# roi_y_start, roi_y_end = 0, 410

# frame_count = 0

# from scipy.stats import mode

# def group_shapes_by_proximity(shapes_dict, proximity_threshold=100):
#     grouped_shapes = {}
#     processed_centroids = set()

#     def is_within_proximity(c1, c2):
#         return abs(c1[0] - c2[0]) <= proximity_threshold and abs(c1[1] - c2[1]) <= proximity_threshold

#     # Iterate over shapes and group by proximity
#     group_id = 1
#     for shape1, data1 in shapes_dict.items():
#         if tuple(data1[1]) in processed_centroids:
#             continue  # Skip already grouped centroids

#         group = [shape1]  # Start a new group with the current shape
#         processed_centroids.add(tuple(data1[1]))

#         # Compare with other shapes
#         for shape2, data2 in shapes_dict.items():
#             if shape1 != shape2 and tuple(data2[1]) not in processed_centroids:
#                 if is_within_proximity(data1[1], data2[1]):
#                     group.append(shape2)
#                     processed_centroids.add(tuple(data2[1]))

#         # Collect shapes, colors, and centroids for the group
#         group_shapes = [s.split('1')[0] for s in group]
#         group_colors = [shapes_dict[s][0] for s in group]
#         group_centroids = [shapes_dict[s][1] for s in group]

#         # Calculate the mode shape, mode color, and mode centroid for the group
#         mode_shape = mode(group_shapes)[0][0]
#         mode_color = mode(group_colors)[0][0]
#         mode_centroid = [
#             int(round(sum(c[0] for c in group_centroids) / len(group_centroids))),
#             int(round(sum(c[1] for c in group_centroids) / len(group_centroids))),
#         ]

#         # Use the mode shape as the group name
#         grouped_shapes[f"{mode_shape.lower()}_{group_id}"] = [mode_color, mode_centroid]
#         group_id += 1

#     return grouped_shapes

# grouped_shapes = {}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Draw a thin outline rectangle for the ROI
#     cv2.rectangle(frame, (roi_x_start, roi_y_start), (roi_x_end, roi_y_end), (0, 255, 255), 1)

#     # Crop the region of interest (ROI)
#     roi = frame[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

#     # Process only the ROI
#     roi_with_rectangles = detect_rectangles(roi, frame, roi_x_start, roi_y_start)
#     roi_with_circles = detect_circles(roi_with_rectangles, frame, roi_x_start, roi_y_start)

#     # Show the frame
#     cv2.imshow("Shapes (Rectangles and Circles)", frame)
    
#     frame_count += 1
#     # After the frame processing loop, group and display the shapes
#     if frame_count == 50:
#         grouped_shapes = group_shapes_by_proximity(shapes_dict)
#         break

#     if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
#         break

# print(grouped_shapes)

# cap.release()
# cv2.destroyAllWindows()
