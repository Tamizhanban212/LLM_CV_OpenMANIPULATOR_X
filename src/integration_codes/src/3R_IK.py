import numpy as np
import matplotlib.pyplot as plt

def ikinematics(link1, link2, link3, positionx, positiony, gamma):
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
    xe = positionx - 0.035  # Adjust for the end-effector offset
    ye = positiony - 0.045  # Adjust for the end-effector offset
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
        J1b = (np.degrees(np.arctan2(y3, x3)) + B)
        J2b = -(180 - a)
        J3b = (gamma - J1b - J2b)

        print(f"The joint angles for elbow-up configuration are: J1, J2, J3 = {90-J1b}, {-(J2b+90)}, {-(J3b)}")
    else:
        print("Dimension error!")
        print("End-effector is outside the workspace.")
        return

    # Compute positions of joints for visualization
    x2b = L12 * np.cos(np.radians(J1b))
    y2b = L12 * np.sin(np.radians(J1b))
    r = L12 + L23 + L34

    # Plot the manipulator configurations
    plt.figure()
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True)

    # Workspace boundary
    circle = plt.Circle((0, 0), r, color='gray', fill=False, linestyle=':')
    plt.gca().add_artist(circle)

    # Elbow-up configuration
    plt.plot([0, x2b, x3, xe], [0, y2b, y3, ye], 'g--', label="Elbow-up")
    plt.plot([0, x2b], [0, y2b], 'go')
    plt.plot([x3], [y3], 'go')
    plt.plot([xe], [ye], 'ro')

    # Axes and labels
    plt.axhline(0, color='red', linewidth=0.5)
    plt.axvline(0, color='red', linewidth=0.5)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Inverse Kinematics 3-Links Planar Manipulator')
    plt.legend()
    plt.show()

# Example usage
if __name__ == "__main__":
    ikinematics(link1=0.13, link2=0.125, link3=0.14, positionx=(0.15**2+0.1**2)**0.5, positiony=0.01, gamma=-90)