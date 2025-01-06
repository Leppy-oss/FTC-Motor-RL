import numpy as np

VOLTAGE_CLIPPING = 50
OBSERVED_TORQUE_LIMIT = 5.7
MOTOR_VOLTAGE = 16.0
MOTOR_RESISTANCE = 0.186
MOTOR_TORQUE_CONSTANT = 0.0954
MOTOR_VISCOUS_DAMPING = 0.01  # Added small damping for realism
MOTOR_SPEED_LIMIT = MOTOR_VOLTAGE / (MOTOR_VISCOUS_DAMPING + MOTOR_TORQUE_CONSTANT)


class MotorModel(object):
    """The accurate motor model, which incorporates motor dynamics and stochasticity."""

    def __init__(self, torque_control_enabled=False, kp=1.2, kd=0, noise_std=0.1):
        self._torque_control_enabled = torque_control_enabled
        self._kp = kp
        self._kd = kd
        self._resistance = MOTOR_RESISTANCE
        self._voltage = MOTOR_VOLTAGE
        self._torque_constant = MOTOR_TORQUE_CONSTANT
        self._viscous_damping = MOTOR_VISCOUS_DAMPING
        self._current_table = [0, 10, 20, 30, 40, 50, 60]
        self._torque_table = [0, 1, 1.9, 2.45, 3.0, 3.25, 3.5]

        # Stochastic noise standard deviation for simulation
        self.noise_std = noise_std

    def set_voltage(self, voltage):
        self._voltage = voltage

    def get_voltage(self):
        return self._voltage

    def set_viscous_damping(self, viscous_damping):
        self._viscous_damping = viscous_damping

    def get_viscous_damping(self):
        return self._viscous_damping

    def convert_to_torque(
        self, motor_commands, current_motor_angle, current_motor_velocity
    ):
        """Convert the commands (position control or torque control) to torque."""
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
        """Convert the PWM signal to torque, including motor dynamics and noise."""

        # Stochasticity: introduce random noise into torque and voltage
        noise_factor = np.random.normal(0, self.noise_std)  # Gaussian noise

        observed_torque = np.clip(
            self._torque_constant * (pwm * self._voltage / self._resistance),
            -OBSERVED_TORQUE_LIMIT,
            OBSERVED_TORQUE_LIMIT,
        )
        observed_torque += noise_factor  # Add noise to the torque

        # Net voltage calculation, including motor dynamics and noise
        voltage_net = np.clip(
            pwm * self._voltage
            - (self._torque_constant + self._viscous_damping) * current_motor_velocity,
            -VOLTAGE_CLIPPING,
            VOLTAGE_CLIPPING,
        )

        # Add random noise to voltage (mimicking sensor fluctuations)
        voltage_net += np.random.normal(0, self.noise_std)

        current = voltage_net / self._resistance
        current_sign = np.sign(current)
        current_magnitude = np.abs(current)

        # Saturate torque based on empirical current relation
        actual_torque = np.interp(
            current_magnitude, self._current_table, self._torque_table
        )
        actual_torque = np.multiply(current_sign, actual_torque)
        actual_torque += noise_factor  # Add noise to the actual torque

        return actual_torque, observed_torque


if __name__ == "__main__":
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
        "5203.urdf",
        basePosition=position,
        baseOrientation=orientation,
        useFixedBase=True,
    )
    joint_index = 1

    target_position = 0.0
    velocity_increment = 2

    while True:
        keys = p.getKeyboardEvents()

        if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_WAS_TRIGGERED:
            target_position += velocity_increment
            print(f"Motor position increased to {target_position:.2f}")

        if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_WAS_TRIGGERED:
            target_position -= velocity_increment
            print(f"Motor position decreased to {target_position:.2f}")

        if ord("q") in keys and keys[ord("q")] & p.KEY_WAS_TRIGGERED:
            print("Exiting...")
            break

        p.setJointMotorControl2(
            bodyIndex=motor,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_position,
        )

        p.stepSimulation()
        time.sleep(0.01)

    p.disconnect()
