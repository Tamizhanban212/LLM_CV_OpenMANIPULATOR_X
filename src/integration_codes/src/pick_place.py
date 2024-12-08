#!/usr/bin/env python3
import efficient_IK as ik
import rospy

def pick_and_place(xfrom, yfrom, xto, yto):
    """
    Generate pick-and-place poses and perform the movements.
    
    Parameters:
        xfrom (float): X-coordinate of the pick position.
        yfrom (float): Y-coordinate of the pick position.
        xto (float): X-coordinate of the place position.
        yto (float): Y-coordinate of the place position.
    """
    # Define joint names
    joint_names = ["joint1", "joint2", "joint3", "joint4"]
    
    # Generate the movement sequence
    movements = [
        (0, 0.15, 0.1, -50, 1.5, "open"),
        (xfrom, yfrom, 0.1, -50, 1.2, "open"),
        (xfrom, yfrom, -0.01, -70, 1.2, "close"),
        (xfrom, yfrom, 0.1, -50, 1.2, "close"),
        (xto, yto, 0.1, -50, 1.5, "close"),
        (xto, yto, 0.04, -70, 1.2, "open"),
        (xto, yto, 0.1, -50, 1.2, "open"),
        (0, 0.15, 0.1, -50, 1.2, "open"),
    ]
    
    # Perform the movements
    for x, y, z, orientation, path_t, gripper_pos in movements:
        try:
            # Calculate joint positions using IK
            joint_positions_degrees = ik.IK_4R(x, y, z, orientation)
            
            # Control the robot joints and gripper
            ik.control_joint(joint_names, joint_positions_degrees, gripper_pos, path_t)
            
            # Wait for the path to complete
            rospy.sleep(path_t)
        except Exception as e:
            rospy.logerr(f"Error performing movement to position ({x}, {y}, {z}): {e}")

if __name__ == "__main__":
    rospy.init_node('joint_control_node', anonymous=True)
    
    # Example pick-and-place coordinates
    xfrom, yfrom = 0.1, 0.1  # Pick position
    xto, yto = 0.1, -0.1      # Place position
    
    # Execute the pick-and-place task
    pick_and_place(xfrom, yfrom, xto, yto)

def switch_off():
    """
    Generate pick-and-place poses and perform the movements.
    
    Parameters:
        xfrom (float): X-coordinate of the pick position.
        yfrom (float): Y-coordinate of the pick position.
        xto (float): X-coordinate of the place position.
        yto (float): Y-coordinate of the place position.
    """
    # Define joint names
    joint_names = ["joint1", "joint2", "joint3", "joint4"]
    
    # Generate the movement sequence
    movements = [
        (0, 0.15, -0.01, -70, 4, "close"),
    ]
    
    # Perform the movements
    for x, y, z, orientation, path_t, gripper_pos in movements:
        try:
            # Calculate joint positions using IK
            joint_positions_degrees = ik.IK_4R(x, y, z, orientation)
            
            # Control the robot joints and gripper
            ik.control_joint(joint_names, joint_positions_degrees, gripper_pos, path_t)
            
            # Wait for the path to complete
            rospy.sleep(path_t)
        except Exception as e:
            rospy.logerr(f"Error performing movement to position ({x}, {y}, {z}): {e}")
