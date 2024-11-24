#!/usr/bin/env python3
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
    return (0, 90, 0)

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
        ((0.1, 0.0, 0.25), "open", 0.2),   # Move to initial hover position
        ((x_from, y_from, 0.15), "open", 0.2),  # Move above the 'from' position
        ((x_from, y_from, 0.06), "open", 0.2),  # Move to the 'from' position
        ((x_from, y_from, 0.06), "close", 0.2), # Close gripper at 'from'
        ((x_from, y_from, 0.12), "close", 0.2), # Lift from the 'from' position
        ((x_to, y_to, 0.15), "close", 0.2),     # Move above the 'to' position
        ((x_to, y_to, 0.07), "close", 0.2),     # Move to the 'to' position
        ((x_to, y_to, 0.07), "open", 0.2),      # Open gripper at 'to'
        ((x_to, y_to, 0.15), "open", 0.2),      # Lift from the 'to' position
        ((0.1, 0.0, 0.25), "open", 0.2),   # Return to hover position
    ]

if __name__ == "__main__":
    rospy.sleep(2)
    # Example pairs of "from" and "to" positions
    pairs = [
        ((0.15, -0.05), (0.15, 0.05)),
        ((0.17, 0.03), (0.15, 0.05)),
    ]
    
    for from_position, to_position in pairs:
        poses = generate_poses(from_position, to_position)
        send_pose(poses)


# import rospy
# from open_manipulator_msgs.srv import SetJointPosition, SetJointPositionRequest
# import math

# def control_gripper(position):
#     rospy.wait_for_service('/goal_tool_control')
#     try:
#         tool_control = rospy.ServiceProxy('/goal_tool_control', SetJointPosition)
#         request = SetJointPositionRequest()
#         request.joint_position.joint_name = ["gripper"]
#         request.joint_position.position = [position]
#         tool_control(request)
#     except rospy.ServiceException as e:
#         rospy.logerr("Service call failed: %s", e)

# def send_joint_positions(joint_positions):
#     rospy.init_node('joint_gripper_control_node')
#     rospy.wait_for_service('/goal_joint_space_path')
#     set_joint_position = rospy.ServiceProxy('/goal_joint_space_path', SetJointPosition)

#     for joint_values_degrees, gripper_state, delay in joint_positions:
#         # Convert degrees to radians
#         joint_values_radians = [math.radians(angle) for angle in joint_values_degrees]

#         try:
#             request = SetJointPositionRequest()
#             request.joint_position.joint_name = ["joint1", "joint2", "joint3", "joint4"]
#             request.joint_position.position = joint_values_radians
#             request.path_time = 2
#             set_joint_position(request)
#             rospy.sleep(delay)
#             if gripper_state != "unchanged":
#                 control_gripper(0.01 if gripper_state == "open" else -0.01)
#         except rospy.ServiceException as e:
#             rospy.logerr("Service call failed: %s", e)
#         rospy.sleep(2)

# if __name__ == "__main__":
#     rospy.sleep(2)
#     joint_positions = [
#         ([0, -45, 35, 0], "open", 0.2),
#     ]
#     send_joint_positions(joint_positions)