import pybullet as p
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO  # Example: PPO algorithm
from motor_model import MotorModel


class MotorControlEnv(gym.Env):
    def __init__(self, motor_model, render=False):
        super(MotorControlEnv, self).__init__()
        self.motor_model = motor_model
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.target_position = 10.0  # Target position (setpoint)
        self.kp = 1.2  # Initial proportional gain
        self.kd = 0.0  # Initial derivative gain
        self.render_flag = render

        # Setup PyBullet
        if self.render_flag:
            self.client = p.connect(p.GUI)  # Connect to PyBullet GUI for visualization
            p.setGravity(0, 0, -9.8)
            self.motor_id = self._create_motor()

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )

    def _create_motor(self):
        # Create a simple motor model for visualization
        motor_id = p.loadURDF(
            "5203.urdf", basePosition=[0, 0, 0.1]
        )  # Placeholder for motor model
        return motor_id

    def step(self, action):
        # Apply action to update the P and D coefficients
        self.kp += action[0]  # Update proportional gain
        self.kd += action[1]  # Update derivative gain

        # Clamp the gains within reasonable limits
        self.kp = np.clip(self.kp, 0.0, 10.0)
        self.kd = np.clip(self.kd, 0.0, 10.0)

        # Simulate motor behavior with the updated P and D
        current_motor_angle = self.current_position
        current_motor_velocity = self.current_velocity
        pwm_signal = (
            -self.kp * (current_motor_angle - self.target_position)
            - self.kd * current_motor_velocity
        )
        pwm_signal = np.clip(pwm_signal, -1.0, 1.0)

        # Update motor state based on the new PWM signal (from the motor model)
        actual_torque, observed_torque = self.motor_model.convert_to_torque(
            pwm_signal, current_motor_angle, current_motor_velocity
        )

        # Simulate the motor's physical response
        self.current_velocity += actual_torque * 0.01  # Simple acceleration model
        self.current_position += self.current_velocity * 0.01

        # Calculate the error and cost (reward function)
        position_error = self.target_position - self.current_position
        velocity_error = self.current_velocity
        cost = position_error**2 + velocity_error**2  # Simplified cost function

        # The reward is the negative cost (minimizing cost is the goal)
        reward = -cost

        # Update PyBullet visualization for motor state
        if self.render_flag:
            self._update_motor_visualization(self.current_position)

        # Check if the episode is done (e.g., target position reached)
        done = False
        if abs(position_error) < 0.1:  # Close enough to target position
            done = True

        # Return the observation (new state), reward, done flag, and any info
        observation = np.array(
            [self.current_position, self.current_velocity, self.kp, self.kd]
        )
        return observation, reward, done, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment for a new episode
        self.current_position = 0.0
        self.current_velocity = 0.0
        self.target_position = np.random.uniform(5.0, 15.0)  # Random target position
        self.kp = 1.2
        self.kd = 0.0
        if self.render_flag:
            self._reset_motor_visualization()
        return np.array(
            [self.current_position, self.current_velocity, self.kp, self.kd]
        )

    def _update_motor_visualization(self, position):
        """Update the motor visualization in PyBullet based on the current position."""
        p.resetBasePositionAndOrientation(
            self.motor_id, [position, 0, 0.1], [0, 0, 0, 1]
        )

    def _reset_motor_visualization(self):
        """Reset the motor visualization to the starting position."""
        p.resetBasePositionAndOrientation(self.motor_id, [0, 0, 0.1], [0, 0, 0, 1])

    def render(self, mode="human"):
        """Render the environment (PyBullet visualization)."""
        if self.render_flag:
            p.stepSimulation()  # Advance the simulation


# Create the motor model with stochasticity
motor_model = MotorModel(torque_control_enabled=True, noise_std=0.1)

# Create the environment with PyBullet rendering
env = MotorControlEnv(motor_model, render=True)

# Define the RL agent (PPO example)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)  # Train for some steps

# Test the trained agent
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, reward, done, info = env.step(action)
    if done:
        break
