#!/usr/bin/env python3
import rospy
from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
import math
import numpy as np

def ikinematics_3R(positionx, positiony, positionz, gamma, link1=0.13, link2=0.125, link3=0.14):
    L12 = link1
    L23 = link2
    L34 = link3
    xe = (positionx**2 + positiony**2)**0.5 - 0.025
    ye = positionz - 0.04
    g = np.radians(gamma)

    x3 = xe - L34 * np.cos(g)
    y3 = ye - L34 * np.sin(g)
    C = np.sqrt(x3**2 + y3**2)

    if (L12 + L23) > C:
        a = np.degrees(np.arccos((L12**2 + L23**2 - C**2) / (2 * L12 * L23)))
        B = np.degrees(np.arccos((L12**2 + C**2 - L23**2) / (2 * L12 * C)))

        J1b = 90 - (np.degrees(np.arctan2(y3, x3)) + B)
        J2b = -(-(180 - a) + 90)
        J3b = -(gamma - (np.degrees(np.arctan2(y3, x3)) + B) + (180 - a))

        return J1b, J2b, J3b
    else:
        print("Dimension error! End-effector is outside the workspace.")
        return None

def IK_4R(x, y, z, orientation):
    J1 = np.degrees(np.arctan2(y, x))
    J2, J3, J4 = ikinematics_3R(x, y, z, orientation)
    return J1, J2, J3, J4

def control_joint(joint_names, positions, gripper_pos, path_t):
    rospy.wait_for_service('/goal_joint_space_path')
    try:
        joint_control = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)
        request = SetJointPositionRequest()
        request.joint_position.joint_name = joint_names
        request.joint_position.position = [math.radians(pos) for pos in positions]
        request.path_time = path_t
        joint_control(request)

        rospy.sleep(path_t)
        tool_control = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
        request = SetJointPositionRequest()
        request.joint_position.joint_name = ["gripper"]
        position = 0.01 if gripper_pos == "open" else -0.01
        request.joint_position.position = [position]
        tool_control(request)
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def transform_pixels(x,y):
    ym = (x - 412)*0.000625
    xm = (y - 40)*0.000625
    return xm, ym

if __name__ == "__main__":
    rospy.init_node('joint_control_node', anonymous=True)
    
    # User-provided list of (x, y, z, orientation, path_t, gripper_pos)
    xm, ym = transform_pixels(535, 319)
    movements = [
        (0, 0.15, 0.07, -80, 1.5, "open"),
        # (0, 0.15, 0.07, -80, 1.5, "close"),
        # (0.2, 0.0, 0, -80, 1.5, "open")
    ]

    joint_names = ["joint1", "joint2", "joint3", "joint4"]

    for x, y, z, orientation, path_t, gripper_pos in movements:
        joint_positions_degrees = IK_4R(x, y, z, orientation)
        control_joint(joint_names, joint_positions_degrees, gripper_pos, path_t)
        rospy.sleep(path_t)

# (0, 0.15, 0.07, -50, 1.5, "open")