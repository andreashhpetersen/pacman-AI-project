import gymnasium as gym
import numpy as np

from layout import Layout
from pacman import Directions, ClassicGameRules
from game import Agent
import random
import game
import util
import textDisplay
from pacman import GameState

from stable_baselines3 import PPO, DQN

class NullAgent:
    def __init__(self):
        pass


action_map = ['Stop', 'North', 'South', 'East', 'West']

class PacmanEnv(gym.Env):

    def __init__(self, layout : Layout, ghosts):
        self.layout = layout
        self.ghosts = ghosts
        self.display = textDisplay.NullGraphics()

        self.ndims = self.layout.width * self.layout.height
        self.observation_space = gym.spaces.Box(
            low=0, high=6,
            shape=(self.layout.width, self.layout.height), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(5)

    def _get_info(self):
        return {}

    def _get_obs(self):
        return np.array(self.game.state.data._stateMap(), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        rules = ClassicGameRules(timeout=30)  # could be parameter
        horizon = -1  # could also be a parameter
        self.game = rules.newGame(
            self.layout, horizon, NullAgent(), self.ghosts, self.display,
            True, False
        )
        self.game.numMoves = 0  # should be set in game object
        self.prevScore = self.game.state.getScore()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = action_map[action]
        reward = 0

        # penalise illegal action and choose always legal 'stop'
        legal = self.game.state.getLegalActions()
        if action not in legal:
            action = 'Stop'
            reward -= 1

        # take a step
        self.game.step(action)

        # calculate reward
        new_score = self.game.state.getScore()
        reward += new_score - self.prevScore
        self.prevScore = new_score

        # return (state, reward, done, truncated, info)
        return self._get_obs(), reward, self.gameOver(), False, self._get_info()

    def gameOver(self):
        return self.game.gameOver


class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, ghosts, **kwargs):
        print(layout)
        self.layout = layout
        self.ghosts = ghosts
        self.offline_planning()


    def offline_planning(self):
        # Compute offline policy and/or value function
        # Time limit: 10 minutes
        model_type = PPO
        if model_type == DQN:
            model_name = './dqn_pacman_smallGrid.zip'
        elif model_type == PPO:
            model_name = './ppo_pacman_smallGrid.zip'
        else:
            raise ValueError(f'model type {model_type} not supported')

        try:
            model = PPO.load(model_name, device='cpu')
        except:
            env = PacmanEnv(self.layout, self.ghosts)
            model = PPO('MlpPolicy', env, verbose=1, device='cpu')
            model.learn(total_timesteps=600_000)
            model.save(model_name)

        self.model = model

    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        obs = np.array(state.data._stateMap(), dtype=np.uint8)
        action, _ = self.model.predict(obs, deterministic=True)
        action = action_map[action]
        if action not in state.getLegalActions():
            action = 'Stop'

        return action
