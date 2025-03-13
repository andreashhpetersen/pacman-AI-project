from layout import Layout
from pacman import Directions
from game import Agent
import random
import game
import util
from pacman import GameState

class PlanningAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def __init__(self, layout : Layout, **kwargs):
        print(layout)
        self.layout = layout
        self.offline_planning()


    def offline_planning(self):
        # Compute offline policy and/or value function
        # Time limit: 10 minutes

        pass


    def getAction(self, state : GameState):
        # Time limit: approx 1 second
        # Look-up offline policy or online search with MCTS/LRTDP using some pre-computed value function?

        print(state.getPacmanState())
        for ghost in state.getGhostStates():
            print(ghost)

        return Directions.STOP
