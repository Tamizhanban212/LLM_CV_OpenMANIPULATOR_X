#!/usr/bin/env python3
import rospy
from open_manipulator_msgs.srv import SetKinematicsPose, SetKinematicsPoseRequest
from geometry_msgs.msg import Pose, Point, Quaternion
import tf.transformations as tf_tr
import math
import speech_recognition as sr
from transformers import pipeline

# Load a Hugging Face pipeline for natural language understanding
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def euler_to_quaternion(roll, pitch, yaw):
    roll = math.radians(roll)
    pitch = math.radians(pitch)
    yaw = math.radians(yaw)
    return Quaternion(*tf_tr.quaternion_from_euler(roll, pitch, yaw))

def compute_orientation(x, y, z):
    if z <= 0.1:
        return (0, 0, 0)
    roll = 0
    pitch = -math.atan2(y, z)
    yaw = math.atan2(x, z)
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))

def send_home_pose():
    rospy.wait_for_service('/goal_task_space_path')
    set_pose = rospy.ServiceProxy('/goal_task_space_path', SetKinematicsPose)
    try:
        x, y, z = 0.1, 0.0, 0.25  # Home position
        orientation = compute_orientation(x, y, z)
        pose = Pose(position=Point(x, y, z), orientation=euler_to_quaternion(*orientation))

        kinematics_pose = SetKinematicsPoseRequest()
        kinematics_pose.kinematics_pose.pose = pose
        kinematics_pose.planning_group = "arm"
        kinematics_pose.end_effector_name = "gripper"
        kinematics_pose.path_time = 2

        rospy.loginfo("Sending manipulator to home position...")
        set_pose(kinematics_pose)
        rospy.loginfo("Manipulator is in home position.")
    except rospy.ServiceException as e:
        rospy.logerr("Service call failed: %s", e)

def recognize_speech_and_respond():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    while not rospy.is_shutdown():
        rospy.loginfo("Listening for command...")
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            try:
                audio = recognizer.listen(source)
                rospy.loginfo("Recognizing speech...")
                speech_text = recognizer.recognize_google(audio)
                rospy.loginfo(f"Recognized: {speech_text}")

                # Use Hugging Face model for command classification
                candidate_labels = ["move to home position", "ignore"]
                result = classifier(speech_text, candidate_labels)
                rospy.loginfo(f"Classification result: {result}")

                if result["labels"][0] == "move to home position" and result["scores"][0] > 0.8:
                    send_home_pose()
                else:
                    rospy.loginfo("No action taken.")
            except sr.UnknownValueError:
                rospy.logwarn("Speech not understood.")
            except sr.RequestError as e:
                rospy.logerr(f"Could not request results from Google Speech Recognition service; {e}")

if __name__ == "__main__":
    rospy.init_node("speech_control_node")
    try:
        recognize_speech_and_respond()
    except rospy.ROSInterruptException:
        pass
