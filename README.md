# Project Title: LLM-CV Integration with OpenMANIPULATOR-X

This project demonstrates the integration of Large Language Models (LLMs) with computer vision (CV) systems to control the OpenMANIPULATOR-X robot. The implementation showcases the use of machine learning for effective robotic manipulations, alongside visual demonstrations.

---

## Table of Contents
1. [Introduction](#introduction)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Features](#features)  
5. [Results](#results)  

---

## Introduction

This project combines advanced Large Language Models and computer vision techniques to interact with the OpenMANIPULATOR-X robot. The primary objective is to demonstrate how LLMs can process and interpret data to execute robotic actions. The system leverages vision-based feedback and pre-trained LLMs to achieve efficient manipulation tasks.

---


---

## Prerequisites

Ensure the following tools and packages are installed on your system:

- **Python 3.8+**  
- **ROS (Robot Operating System)**  
- **Dependencies for OpenMANIPULATOR-X:**  
  - Dynamixel SDK  
  - OpenMANIPULATOR-X ROS packages  
- **Required Python Libraries:**  
  - TensorFlow / PyTorch (for LLM implementation)  
  - OpenCV (for computer vision)  
  - NumPy  
  - Other relevant dependencies mentioned in `requirements.txt`  

---

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/IITGN-Robotics/project-submission-3-bonus-the_roboticists.git
   cd project-submission-3-bonus-the_roboticists
2. Install the required Python packages:
   ```bash
    pip install -r requirements.txt
3. Set up the ROS workspace:
   ```bash
    cd ~/project-submission-3-bonus-the_roboticists
    catkin_make
    source devel/setup.bash
4. Ensure the OpenMANIPULATOR-X is connected and configured.
   ```bash
    cd ~/project-submission-3-bonus-the_roboticists/src/llm_codes/src/
    python3 Final_code.py
   
## Features

- **LLM Integration**: Utilizes pre-trained language models for generating robotic commands.  
- **Computer Vision**: Real-time object detection and localization using OpenCV.  
- **OpenMANIPULATOR-X Control**: Efficient task execution using motion planning and inverse kinematics.  
- **Modular Code**: Clean and reusable code structure.  

## Results

The project was tested successfully, and the OpenMANIPULATOR-X executed complex tasks effectively using LLM-generated instructions.  
A video demonstration of the working system can be found in **LLM_CV_working_video.mp4**.

