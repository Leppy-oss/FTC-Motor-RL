import pybullet as p
import pybullet_data
import numpy as np
import time
from motor_model import MotorModel

VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)


class MotorModel(object):
    """The accurate motor model, which is based on the physics of DC motors.

    The motor model support two types of control: position control and torque
    control. In position control mode, a desired motor angle is specified, and a
    torque is computed based on the internal motor model. When the torque control
    is specified, a pwm signal in the range of [-1.0, 1.0] is converted to the
    torque.

    The internal motor model takes the following factors into consideration:
    pd gains, viscous friction, back-EMF voltage and current-torque profile.
    """

    def __init__(self, torque_control_enabled=False, kp=1.2, kd=0):
        self._torque_control_enabled = torque_control_enabled
        self._kp = kp
        self._kd = kd
        self._resistance = MOTOR_RESISTANCE
        self._voltage = MOTOR_VOLTAGE
        self._torque_constant = MOTOR_TORQUE_CONSTANT
        self._viscous_damping = MOTOR_VISCOUS_DAMPING
        self._current_table = [0, 10, 20, 30, 40, 50, 60]
        self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]

    def set_voltage(self, voltage):
        self._voltage = voltage

    def get_voltage(self):
        return self._voltage

    def set_viscous_damping(self, viscous_damping):
        self._viscous_damping = viscous_damping

    def get_viscous_dampling(self):
        return self._viscous_damping

    def convert_to_torque(
        self, motor_commands, current_motor_angle, current_motor_velocity
    ):
        """Convert the commands (position control or torque control) to torque.

        Args:
          motor_commands: The desired motor angle if the motor is in position
            control mode. The pwm signal if the motor is in torque control mode.
          current_motor_angle: The motor angle at the current time step.
          current_motor_velocity: The motor velocity at the current time step.
        Returns:
          actual_torque: The torque that needs to be applied to the motor.
          observed_torque: The torque observed by the sensor.
        """
        if self._torque_control_enabled:
            pwm = motor_commands
        else:
            pwm = (
                -self._kp * (current_motor_angle - motor_commands)
                - self._kd * current_motor_velocity
            )
        pwm = np.clip(pwm, -1.0, 1.0)
        return self._convert_to_torque_from_pwm(pwm, current_motor_velocity)

    def _convert_to_torque_from_pwm(self, pwm, current_motor_velocity):
        """Convert the pwm signal to torque.

        Args:
          pwm: The pulse width modulation.
          current_motor_velocity: The motor velocity at the current time step.
        Returns:
          actual_torque: The torque that needs to be applied to the motor.
          observed_torque: The torque observed by the sensor.
        """
        observed_torque = np.clip(
            self._torque_constant * (pwm * self._voltage / self._resistance),
            -OBSERVED_TORQUE_LIMIT,
            OBSERVED_TORQUE_LIMIT,
        )

        # Net voltage is clipped at VOLTAGE_CLIPPING by diodes on the motor controller.
        voltage_net = np.clip(
            pwm * self._voltage
            - (self._torque_constant + self._viscous_damping) * current_motor_velocity,
            -VOLTAGE_CLIPPING,
            VOLTAGE_CLIPPING,
        )
        current = voltage_net / self._resistance
        current_sign = np.sign(current)
        current_magnitude = np.absolute(current)

        # Saturate torque based on empirical current relation.
        actual_torque = np.interp(
            current_magnitude, self._current_table, self._torque_table
        )
        actual_torque = np.multiply(current_sign, actual_torque)
        return actual_torque, observed_torque


# Simulation with velocity control
def main():
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
        "5203.urdf",
        basePosition=position,
        baseOrientation=orientation,
        useFixedBase=True,
    )
    joint_index = 1  # Assuming the motor joint is index 0

    # Initialize motor model
    motor_model = MotorModel()

    # Simulation parameters
    target_pos = 5.0  # Desired speed in rad/s
    time_step = 1.0 / 240.0  # PyBullet default time step

    # Main simulation loop
    while True:
        # Get current motor velocity
        joint_state = p.getJointState(motor, joint_index)
        current_velocity = joint_state[1]  # Joint velocity
        print(f"Current velocity: {current_velocity:.2f}")

        # Compute desired speed using the motor model
        speed_command = motor_model.compute_speed(target_pos)

        # Apply velocity control to the motor joint
        p.setJointMotorControl2(
            bodyIndex=motor,
            jointIndex=joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed_command,
            force=10.0,  # Maximum force/torque the motor can exert
        )

        # Step simulation
        p.stepSimulation()
        time.sleep(time_step)


# Run the simulation
if __name__ == "__main__":
    main()
