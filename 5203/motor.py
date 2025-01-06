import gymnasium as gym
from gymnasium import spaces
import pybullet as p
import pybullet_data
import numpy as np


class MotorControlEnv(gym.Env):
    def __init__(self):
        super(MotorControlEnv, self).__init__()
        # Connect to PyBullet
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load URDF
        self.motor = p.loadURDF("5203.urdf", useFixedBase=True)
        self.joint_index = 0  # Assuming the motor joint is index 0

        # Observation and action spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self.target_position = 0.0
        self.dt = 0.01  # Simulation time step

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        speed_command = action[0]

        # Apply speed command
        p.setJointMotorControl2(
            bodyIndex=self.motor,
            jointIndex=self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=speed_command,
        )

        # Step simulation
        p.stepSimulation()

        # Observe
        joint_state = p.getJointState(self.motor, self.joint_index)
        position, velocity = joint_state[0], joint_state[1]
        error = self.target_position - position

        # Calculate reward
        reward = -abs(error) - 0.01 * abs(velocity)

        # Done condition
        done = abs(error) < 0.01

        observation = np.array(
            [position, velocity, self.target_position], dtype=np.float32
        )
        return observation, reward, done, {}

    def reset(self):
        p.resetSimulation(self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.motor = p.loadURDF("5203.urdf", useFixedBase=True)
        self.target_position = np.random.uniform(-np.pi, np.pi)
        p.setJointMotorControl2(
            bodyIndex=self.motor,
            jointIndex=self.joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
        )
        return np.array([0.0, 0.0, self.target_position], dtype=np.float32)

    def render(self, mode="human"):
        pass

    def close(self):
        p.disconnect(self.client)
