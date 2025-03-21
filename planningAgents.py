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
    WALL = 0
    EMPTY = 1
    FOOD = 2
    CAPSULE = 3
    GHOST = 4
    PACMAN = 5


    def __init__(self, layout : Layout, ghosts):
        self.layout = layout
        self.ghosts = ghosts
        self.display = textDisplay.NullGraphics()

        self.ndims = self.layout.width * self.layout.height
        self.observation_space = gym.spaces.Box(
            low=0, high=5,
            shape=(self.layout.width, self.layout.height), dtype=np.uint8
        )
        self.action_space = gym.spaces.Discrete(5)

    def _get_info(self):
        return {}

    def _get_obs(self):
        sdata = self.game.state.data

        grid = np.ones((self.layout.width, self.layout.height), dtype=np.uint8)
        grid[sdata.food.data] = self.FOOD
        grid[sdata.layout.walls.data] = self.WALL

        if len(sdata.capsules) > 0:
            grid[tuple(zip(*sdata.capsules))] = self.CAPSULE

        for agentState in sdata.agentStates:
            if agentState == None or agentState.configuration is None:
                continue
            x, y = map(int, agentState.configuration.pos)
            grid[x][y] = self.PACMAN if agentState.isPacman else self.GHOST

        return grid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.rules = ClassicGameRules(timeout=30)  # could be parameter
        horizon = -1  # could also be a parameter
        self.game = self.rules.newGame(
            self.layout, horizon, NullAgent(), self.ghosts, self.display,
            True, False
        )
        self.game.numMoves = 0  # should be set in game object
        self.prevScore = self.game.state.getScore()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = action_map[action]
        reward = 0
        game = self.game

        # penalise illegal action and choose always legal 'stop'
        legal = game.state.getLegalActions()
        if action not in legal:
            # action = legal[np.random.randint(0, len(legal))]
            action = 'Stop'
            reward -= 2

        # take a step
        agent_idx = 0  # start with Pacman
        while not game.gameOver and agent_idx < len(game.agents):

            # move ghosts
            if agent_idx != 0:
                action = game.agents[agent_idx].getAction(game.state)

            # execute the action
            game.moveHistory.append((agent_idx, action))
            game.state = game.state.generateSuccessor(agent_idx, action)

            # change the display
            self.display.update(game.state.data)

            # check game specific conditions (winning, losing, etc.)
            self.rules.process(game.state, game)

            agent_idx += 1

        # track progress
        game.numMoves += 1

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
            model_name = './ppo_pacman_mediumClassic_6_000_000ts.zip'
        else:
            raise ValueError(f'model type {model_type} not supported')

        try:
            model = PPO.load(model_name, device='cpu')
        except:
            env = PacmanEnv(self.layout, self.ghosts)
            model = PPO('MlpPolicy', env, verbose=1, device='cpu')
            # model = PPO.load(model_name, env, device='cpu')
            model.learn(total_timesteps=3_000_000)
            model.save(model_name)

        self.model = model

    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        obs = state.data._stateMap()
        action, _ = self.model.predict(obs, deterministic=True)
        action = action_map[action]
        if action not in state.getLegalActions():
            action = 'Stop'

        return action
