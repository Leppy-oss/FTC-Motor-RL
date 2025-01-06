import time
from typing import Tuple, Optional, List, Type

import pygame
import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

from gymnasium import ObservationWrapper
from gymnasium.wrappers import TimeLimit
from gymnasium.spaces import Box

from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.policies import BaseModel
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import update_learning_rate
from stable_baselines3.td3.policies import Actor, TD3Policy

from gym_electric_motor import gym_electric_motor as gem
from gym_electric_motor.physical_systems.mechanical_loads import ConstantSpeedLoad
from gym_electric_motor.physical_system_wrappers import CosSinProcessor, DeadTimeProcessor, DqToAbcActionProcessor
from gym_electric_motor.envs.motors import ActionType, ControlType, Motor, MotorType
from gym_electric_motor.physical_systems.solvers import EulerSolver

class FeatureWrapper(ObservationWrapper):
    """
    Wrapper class which wraps the environment to change its observation from a tuple to a flat vector.
    """

    def __init__(self, env):
        """
        Changes the observation space from a tuple to a flat vector
        
        Args:
            env(GEM env): GEM environment to wrap
        """
        super(FeatureWrapper, self).__init__(env)
        state_space = self.env.observation_space[0]
        ref_space = self.env.observation_space[1]
        
        new_low = np.concatenate((state_space.low,
                                  ref_space.low))
        new_high = np.concatenate((state_space.high,
                                   ref_space.high))

        self.observation_space = Box(new_low, new_high)

    def observation(self, observation):
        """
        Gets called at each return of an observation.
        
        """
        observation = np.concatenate((observation[0],
                                      observation[1],
                                      ))
        return observation
    
class LastActionWrapper(gym.Wrapper):
    def __init__(self, env):
        super(LastActionWrapper, self).__init__(env)
        state_space = self.env.observation_space
        action_space = self.env.action_space
        
        new_low = np.concatenate((state_space.low,
                                  action_space.low))
        new_high = np.concatenate((state_space.high,
                                   action_space.high))

        self.observation_space = Box(new_low, new_high)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.last_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        return np.concatenate((observation, self.last_action)), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.last_action = action
        return np.concatenate((observation, self.last_action)), reward, terminated, truncated, info

class VisualizedPMSMEnv(gym.Env):
    def __init__(self, base_env):
        self.base_env = base_env
        self.observation_space = base_env.observation_space
        self.action_space = base_env.action_space

        pygame.init()
        self.screen = pygame.display.set_mode((400, 400))
        pygame.display.set_caption("Motor Visualization")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        self.angle = 0 # Motor shaft angle for visualization

    def step(self, action):
        obs, reward, terminated, truncated, info = self.base_env.step(action)

        # Update visualization variables
        self.angle += action[0] * 10  # Scale action to update shaft angle
        self.angle %= 360

        # Draw visualization
        self._draw(obs)

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.base_env.reset()
        self.angle = 0
        self._draw(obs)
        return obs

    def _draw(self, obs):
        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw motor shaft
        center = (200, 200)
        radius = 50
        end_x = int(center[0] + radius * np.cos(np.radians(self.angle)))
        end_y = int(center[1] + radius * np.sin(np.radians(self.angle)))
        pygame.draw.line(self.screen, (0, 0, 0), center, (end_x, end_y), 5)
        pygame.draw.circle(self.screen, (0, 0, 0), center, 5)

        # Draw text (e.g., angular velocity)
        # angular_velocity = angular_velocity = obs[0][0] if isinstance(obs[0], (tuple, list, np.ndarray)) else obs[0]
        # text_surface = self.font.render(f"Angular Velocity: {angular_velocity:.2f}", True, (0, 0, 0))
        # self.screen.blit(text_surface, (10, 10))

        # Update display
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        self.base_env.close()
        pygame.display.quit()
        pygame.quit()

motor_params = dict(
    p=3,                # [p] = 1, nb of pole pairs
    r_s=17.932e-3,      # [r_s] = Ohm, stator resistance
    l_d=0.37e-3,        # [l_d] = H, d-axis inductance
    l_q=1.2e-3,         # [l_q] = H, q-axis inductance
    psi_p=65.65e-3,     # [psi_p] = Vs, magnetic flux of the permanent magnet
)

nominal_values = dict(
    omega=6000 * 2 * np.pi / 60,  # angular velocity in rad/s
    i=240,                        # motor current in amps
    u=350,                        # nominal voltage in volts
)

limit_values = dict(
    omega=6000 * 2 * np.pi / 60,
    i=240 * 1.2,
    u=350,
)

pmsm_init = {
    'states': {
        'i_sd': 0.0,
        'i_sq': 0.0,
        'epsilon': 0.0,
    }
}

physical_system_wrappers = [
    CosSinProcessor(angle='epsilon'),
    DqToAbcActionProcessor.make('PMSM'),
    DeadTimeProcessor(steps=1)
]

load = ConstantSpeedLoad(omega_fixed=4000 * 2 * np.pi / 60 * 0.2)

env = gem.make(
    'Cont-SC-PMSM-v0',
    motor=dict(
        motor_parameter=motor_params,
        limit_values=limit_values,
        nominal_values=nominal_values,
        motor_initializer=pmsm_init,
    ),
    load=load,
    tau=1e-4,  # Sampling time
    ode_solver=EulerSolver(),
    physical_system_wrappers=physical_system_wrappers,
    state_filter=["i_sd", "i_sq", "omega", "epsilon", "sin(epsilon)", "cos(epsilon)"],
    supply=dict(u_nominal=350),
)

eps_idx = env.unwrapped.physical_system.state_names.index('epsilon')
i_sd_idx = env.unwrapped.physical_system.state_names.index('i_sd')
i_sq_idx = env.unwrapped.physical_system.state_names.index('i_sq')

env = VisualizedPMSMEnv(env)

env = TimeLimit(LastActionWrapper(FeatureWrapper(env)), max_episode_steps=200)

print(env.action_space.sample())

class CustomDDPG(DDPG):
    def __init__(self, policy, env, *args, actor_lr=1e-5, critic_lr=1e-4, **kwargs):
        super().__init__(policy, env, *args, **kwargs)
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

    def _update_learning_rate(self, optimizers):
        """
                Costum function to update actor and critic with different learning rates.
                Based on https://github.com/DLR-RM/stable-baselines3/issues/338
                """
        actor_optimizer, critic_optimizer = optimizers

        update_learning_rate(actor_optimizer, self.actor_lr)
        update_learning_rate(critic_optimizer, self.critic_lr)

def create_network(input_dim, hidden_sizes, output_dim, activations):
    layers = []
    
    # Input layer
    layers.append(nn.Linear(input_dim, hidden_sizes[0]))
    
    # Activation function for first hidden layer
    if activations[0] is not None:
        activation, *params = activations[0]
        act_func = getattr(nn, activation)(*params)
        layers.append(act_func)

    # Hidden layers
    for i in range(1, len(hidden_sizes)):
        layers.append(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))

        # Activation function
        if activations[i] is not None:
            activation, *params = activations[i]
            act_func = getattr(nn, activation)(*params)
            layers.append(act_func)

    # Output layer
    layers.append(nn.Linear(hidden_sizes[-1], output_dim))

    # Activation function for output layer
    if activations[-1] is not None:
        activation, *params = activations[-1]
        act_func = getattr(nn, activation)(*params)
        layers.append(act_func)

    return nn.Sequential(*layers)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# Tanh necessary to "squash the output"
actor_activations = [('ReLU',), ('LeakyReLU', 0.2), ('Tanh',)]
critic_activations = [('ReLU',), ('LeakyReLU', 0.1), None]

actor_network = create_network(
    input_dim = state_dim,
    output_dim = action_dim,
    hidden_sizes =[100, 100],
    activations=actor_activations)

critic_network = create_network(
    input_dim=state_dim + action_dim,
    output_dim=1,
    hidden_sizes= [100, 100],
    activations=critic_activations)

class CustomActor(Actor):
    """
    Actor network (policy) for TD3.
    """
    def __init__(self, *args, **kwargs):
        super(CustomActor, self).__init__(*args, **kwargs)
        # self.mu = nn.Sequential(...)
        self.mu = actor_network

class CustomContinuousCritic(BaseModel):
    """
    Critic network(s) for DDPG/SAC/TD3.
    """
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        action_dim = get_action_dim(self.action_space)

        self.share_features_extractor = share_features_extractor
        self.n_critics = n_critics
        self.q_networks = []
        for idx in range(n_critics):
            # Critic with dropout
            q_net = critic_network
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, ...]:
        # Learn the features extractor with policy loss only
        with th.set_grad_enabled(not self.share_features_extractor):
            features = self.extract_features(obs, self.features_extractor)
        qvalue_input = th.cat([features, actions], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def q1_forward(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """
        Only predict the Q-value using the first network.
        This allows to reduce computation when all the estimates are not needed
        (e.g. when updating the policy in TD3).
        """
        with th.no_grad():
            features = self.extract_features(obs, self.features_extractor)
        return self.q_networks[0](th.cat([features, actions], dim=1))

class CustomTD3Policy(TD3Policy):
    def __init__(self, *args, **kwargs):

        
        super(CustomTD3Policy, self).__init__(*args, **kwargs)


    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomActor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return CustomActor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> CustomContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        return CustomContinuousCritic(**critic_kwargs).to(self.device)

TD3.policy_aliases["CustomTD3Policy"] = CustomTD3Policy

nb_steps = 2560 # number of training steps
buffer_size = nb_steps #number of old observation steps saved
learning_starts = 32 # memory warmup
train_freq = 1 # prediction network gets an update each train_freq's step
batch_size = 32 # mini batch size drawn at each update step
gamma = 0.85
verbose = 1 # verbosity of stable-basline's prints
lr_actor = 1e-4
lr_critic = 1e-3

policy_kwargs = dict(optimizer_class=th.optim.Adam,)

model = CustomDDPG("CustomTD3Policy", env, buffer_size=buffer_size, learning_starts=learning_starts ,train_freq=train_freq, 
            batch_size=batch_size, gamma=gamma, policy_kwargs=policy_kwargs, 
            verbose=verbose, actor_lr=lr_actor, critic_lr=lr_critic)


start_time = time.time()
model.learn(total_timesteps=nb_steps)
total_time = time.time() - start_time
print(f"Batch size 1 total time: {total_time // 60} minutes, {total_time % 60} seconds")

model.save("ddpg_gobilda_5203")