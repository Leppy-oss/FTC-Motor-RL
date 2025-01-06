import pybullet as p
import time


# Connect to PyBullet
p.connect(p.GUI)

camera_distance = 0.25  # Smaller value means more zoomed in
camera_yaw = 45  # Angle to rotate around the Z-axis
camera_pitch = -30  # Angle to tilt up or down
camera_target_position = [0, 0, -0.05]  # Focus point of the camera

p.resetDebugVisualizerCamera(
    cameraDistance=camera_distance,
    cameraYaw=camera_yaw,
    cameraPitch=camera_pitch,
    cameraTargetPosition=camera_target_position,
)

# Load the motor URDF
position = [0, 0, 0]
orientation = p.getQuaternionFromEuler([0, 0, 0])
motor = p.loadURDF(
    "5203.urdf", basePosition=position, baseOrientation=orientation, useFixedBase=True
)
joint_index = 1  # Assuming the motor joint is index 0

# Initial motor velocity
motor_velocity = 0.0
velocity_increment = 0.2  # Change in velocity per key press

print("Keyboard Controls:")
print("UP: Increase velocity")
print("DOWN: Decrease velocity")
print("Q: Quit")

while True:
    keys = p.getKeyboardEvents()

    # Check for key presses
    if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
        motor_velocity += velocity_increment
        print(f"Motor velocity increased to {motor_velocity:.2f}")

    if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED:
        motor_velocity -= velocity_increment
        print(f"Motor velocity decreased to {motor_velocity:.2f}")

    if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
        print("Exiting...")
        break

    # Apply motor velocity
    p.setJointMotorControl2(
        bodyIndex=motor,
        jointIndex=joint_index,
        controlMode=p.VELOCITY_CONTROL,
        targetVelocity=motor_velocity,
    )

    # Step simulation
    p.stepSimulation()
    time.sleep(0.01)  # To simulate real-time behavior

# Disconnect from PyBullet
p.disconnect()
