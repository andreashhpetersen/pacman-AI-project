import gymnasium as gym
import numpy as np

import time
from layout import Layout
from pacman import Directions, ClassicGameRules, GhostRules
from game import Agent
import random
import game
import util
import textDisplay
import itertools
from pacman import GameState

from stable_baselines3 import PPO, DQN

class NullAgent:
    def __init__(self):
        pass


action_map = ['Stop', 'North', 'South', 'East', 'West']

class AbstractState:
    def __init__(self, state: GameState):
        self.agentStates = state.data.copyAgentStates(state.data.agentStates)
        self.capsules = state.getCapsules()     # The invincibility powerups. For good measure.

    def write_to(self, state: GameState):
        state.data.agentStates = state.data.copyAgentStates(self.agentStates)
        state.data.capsules = self.capsules[:]

        return state

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
        self.gamestate = None # Dummy game-state that will be overwritten by abstract states to save deepcopies.
        self.k_lookahead = 2

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
        self.gamestate = self.game.state.deepCopy()
        self.game.numMoves = 0  # should be set in game object
        self.prevScore = self.game.state.getScore()

        return self._get_obs(), self._get_info()

    def step(self, action):
        action = action_map[action]
        reward = 0
        game = self.game

        # Apply shield 🛡️
        suggested = action
        action = self.lookahead_shield(action, self.game.state)
        # Penalise illegal action.
        if suggested != action:
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

    def _any_ghost_action(self, initial_state, n_ghosts):
        successors = []
        ghost_actions = []
        for ghost in range(1, n_ghosts + 1):
            ghost_actions.append(GhostRules.getLegalActions(initial_state, ghost))
        
        initial_stateʹ = AbstractState(initial_state)

        if self.gamestate == None:
            self.gamestate = initial_state.deepCopy()

        assert(len(ghost_actions[0]) > 0)
        for ghost_action in itertools.product(*ghost_actions):
            successor = initial_stateʹ.write_to(self.gamestate) # Overwrite the dummy game-state with the abstract state
            for (ghost, ghost_action) in enumerate(ghost_action):
                ghost = ghost + 1 # pacman is 0.
                successor = successor.generateSuccessor(ghost, ghost_action)
                if successor.isLose():
                    return None
            successors.append(AbstractState(successor))
        
        return successors
    

    # Shield with k-step lookahead.
    def lookahead_shield(self, suggested_action: str, state: GameState):
        legal = state.getLegalActions()
        allowed = []
        time.sleep(0.1)
        pacman = 0
        n_ghosts = state.getNumAgents() - 1

        #print()
        #print()
        #print("🛡️ Sheilding " + suggested_action)
        # Actual lookahead loop. 
        for actionʹ in legal:
            #print("  Checking " + actionʹ)
            action_safe = True

            # Call to generateSucessor actually mutates `state`, so the new assignment is purely for clarity.
            after_action = state.generateSuccessor(pacman, actionʹ) 

            if after_action.isLose():
                #print("    Judgement: Immediate death.")
                continue
            
            sucessors = self._any_ghost_action(after_action, n_ghosts)

            assert(sucessors == None or len(sucessors) > 0)

            # TODO: Add sucessors to queue

            if sucessors == None:
                #print("    Judgement: Ghost might catch you.")
                pass
            else:
                allowed.append(actionʹ)
                #print("    Judgement: Safe :-)")

        #print()

        # Return suggested action, or alternate action if the suggested is not allowed.
        if suggested_action not in allowed:
            if 'Stop' in allowed:
                action = 'Stop'
            elif len(allowed) == 0:
                #print("    Hail Mary! 🙏")
                if suggested_action in legal:
                    action = suggested_action
                else:
                    action = legal[0]
            else:
                action = allowed[0]
        else: 
            action = suggested_action

        #print("  Picked 👉 " + action)
        return action


class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, ghosts, layout_name, **kwargs):
        self.layout = layout
        self.ghosts = ghosts
        self.env = PacmanEnv(self.layout, self.ghosts)
        self.layout_name = layout_name
        self.train_steps = 3000
        #print(layout)
        #print("Training for " + str(self.train_steps) + " steps.")
        self.offline_planning()


    def offline_planning(self):
        # Compute offline policy and/or value function
        # Time limit: 10 minutes
        model_type = PPO
        if model_type == DQN:
            model_name = './dqn_pacman_smallGrid.zip'
        elif model_type == PPO:
            model_name = './ppo_pacman_' + self.layout_name + str(self.train_steps) + '.zip'
        else:
            raise ValueError(f'model type {model_type} not supported')

        try:
            model = PPO.load(model_name, device='cpu')
        except:
            env = self.env
            model = PPO('MlpPolicy', env, verbose=1, device='cpu')
            model.learn(total_timesteps=self.train_steps)
            model.save(model_name)

        self.model = model

    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        obs = state.data._stateMap()
        action, _ = self.model.predict(obs, deterministic=True)
        action = action_map[action]
        action = self.env.lookahead_shield(action, state)

        return action
