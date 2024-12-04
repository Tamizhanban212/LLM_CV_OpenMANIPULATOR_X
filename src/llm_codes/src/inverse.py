import numpy as np

def ikinematics_3R(positionx, positiony, positionz, gamma, link1=0.13, link2=0.125, link3=0.14):
    """
    Inverse kinematics for 3-Link Planar Manipulator
    Inputs:
    - link1, link2, link3: Lengths of the links in the robotic arm
    - positionx, positiony: End-effector coordinates (x, y)
    - gamma: Orientation angle of the end-effector in degrees

    Outputs:
    - Joint angles for elbow-down and elbow-up configurations
    """
    # Link lengths and end-effector position
    L12 = link1
    L23 = link2
    L34 = link3
    xe = (positionx**2+positiony**2)**0.5 - 0.035  # Adjust for the end-effector offset
    ye = positionz - 0.045  # Adjust for the end-effector offset
    g = np.radians(gamma)  # Convert orientation to radians

    # Compute position of joint P3
    x3 = xe - L34 * np.cos(g)
    y3 = ye - L34 * np.sin(g)
    C = np.sqrt(x3**2 + y3**2)

    if (L12 + L23) > C:  # Check if the end-effector is within reach
        # Angles a and B
        a = np.degrees(np.arccos((L12**2 + L23**2 - C**2) / (2 * L12 * L23)))
        B = np.degrees(np.arccos((L12**2 + C**2 - L23**2) / (2 * L12 * C)))

        # Joint angles for elbow-up configuration
        J1b = 90-(np.degrees(np.arctan2(y3, x3)) + B)
        J2b = -(-(180 - a)+90)
        J3b = -(gamma - (np.degrees(np.arctan2(y3, x3)) + B) + (180 - a))

        return J1b, J2b, J3b
    else:
        print("Dimension error!")
        print("End-effector is outside the workspace.")
        return None

def IK_4R(x,y,z, orientation):
    J1 = np.degrees(np.arctan2(y,x))
    J2, J3, J4 = ikinematics_3R(x, y, z, orientation)
    return J1, J2, J3, J4

# Example usage
if __name__ == "__main__":
    # End-effector position and orientation
    [x, y, z, orientation] = [0.1, 0.1, 0.01,-80]
    J1, J2, J3, J4 = IK_4R(x,y,z,orientation)
    print(f"Joint angles for elbow down/up configuration: {J1}, {J2}, {J3}, {J4}")
