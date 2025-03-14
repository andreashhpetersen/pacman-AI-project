Scrips for the assignment of creating a planning/rl agent for Pacman. Taken from an assignment at (https://ai.berkeley.edu/project_overview.html)

To run the game with some of the layouts that we will use for evaluation use: 

`python pacman.py -p PlanningAgent -l knownSmall `
or
`python pacman.py -p PlanningAgent -l knownMedium `
or
`python pacman.py -p PlanningAgent -l knownLarge`

The code of the planning agent is in the file planningAgents.py. Please, rename the agent to give it the name you want (simply by changing the name of the class), then replace PlanningAgent above with the name of the class of your agent. You can also rename the file (add some other file other than planningAgents.py) but it should end in Agents.py for the game to load your agent.

Your task is to implement the offline_planning function (where you precompute some V or Q function in some way, e.g. using RL, or using ideas based on Pattern Databases) and/or policy function; and the getAction function where you lookup your policy and/or perform online search (e.g., MCTS, LRTDP). 

Any questions, contact alto@cs.aau.dk


-------------
# Licensing Information

You are free to use or extend these projects for educational purposes provided that (1)
you do not distribute or publish solutions, (2) you retain this notice, and (3) you
provide clear attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

# Attribution Information

The Pacman AI projects were developed at UC Berkeley.  The core projects and autograders
were primarily created by John DeNero (denero@cs.berkeley.edu) and Dan Klein
(klein@cs.berkeley.edu).  Student side autograding was added by Brad Miller, Nick Hay, and
Pieter Abbeel (pabbeel@cs.berkeley.edu).
