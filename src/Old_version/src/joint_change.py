#!/usr/bin/env python3
import rospy
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import math  # For degree-to-radian conversion
import time  # For sleep function

def control_joint(joint_names, positions, gripper_pos, path_t):
    rospy.wait_for_service('/goal_joint_space_path')
    try:
        joint_control = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        request = SetJointPositionRequest()
        request.joint_position.joint_name = joint_names
        # Convert degrees to radians
        request.joint_position.position = [math.radians(pos) for pos in positions]
        request.path_time = path_t
        joint_control(request)

        tool_control = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        if gripper_pos == "open":
            position = 0.01
        else:
            position = -0.01
        time.sleep(path_t)
        request.joint_position.position = [position]
        tool_control(request)

    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

if __name__ == "__main__":
    rospy.init_node('joint_control_node', anonymous=True)
    
    # Example: Moving joints to specific positions
    joint_names = ["joint1", "joint2", "joint3", "joint4"]
    # Input positions in degrees
    joint_positions_degrees = [45, -18.481512800810506, 27.88455493434809, 30] # Degrees for joints
    path_time = 1  # Time to move to the positions

    control_joint(joint_names, joint_positions_degrees, "open", path_time)
    rospy.sleep(path_time)  # Wait for the movement to complete
