from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def __init__(self):
        self.lastPositions = []
        self.dc = None


def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument 
    is an object of GameState class. Following are a few of the helper methods that you 
    can use to query a GameState object to gather information about the present state 
    of Pac-Man, the ghosts and the maze.
    
    gameState.getLegalActions(): 
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action): 
        Returns the successor state after the specified agent takes the action. 
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.direction gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game
        It corresponds to Utility(s)

    
    The GameState class is defined in pacman.py and you might want to look into that for 
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best


    return legalMoves[chosenIndex]

def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

######################################################################################
# Problem 1a: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (problem 1)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction. Terminal states can be found by one of the following:
          pacman won, pacman lost or there are no legal moves.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          Directions.STOP:
            The stop direction, which is always legal

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game

          gameState.getScore():
            Returns the score corresponding to the current state of the game
            It corresponds to Utility(s)

          gameState.isWin():
            Returns True if it's a winning state

          gameState.isLose():
            Returns True if it's a losing state

          self.depth:
            The depth to which search should continue
        """

        # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
        def successor(state, agent, next_agent, depth, action):
            is_player = next_agent == 0
            obj, opp = max if is_player else min, min if is_player else max
            return estimate(state.generateSuccessor(agent, action), next_agent, obj, opp, depth)[0], action

        def estimate(state, agent, objective, opposite, depth):
            if state.isLose() or state.isWin() or depth <= 0:
                return self.evaluationFunction(state), None

            next_agent, depth = (0, depth - 1) if (agent + 1) == state.getNumAgents() else (agent + 1, depth)
            next_state = (opposite(float('inf'), float('-inf')), Directions.STOP)

            for action in state.getLegalActions(agent):
                candidate = successor(state, agent, next_agent, depth, action)
                next_state = candidate if objective(candidate[0], next_state[0]) == candidate[0] and candidate[0] != next_state[0] else next_state
            return next_state

        return estimate(gameState, 0, max, min, self.depth)[1]
        # END_YOUR_ANSWER

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (problem 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """

        # BEGIN_YOUR_ANSWER (our solution is 42 lines of code, but don't worry if you deviate from this)
        def alpha_cut(alpha, beta, value):
            return max(alpha, value), beta, value > beta

        def beta_cut(alpha, beta, value):
            return alpha, min(beta, value), value < alpha

        def successor(state, agent, next_agent, depth, alpha, beta, action):
            is_player = next_agent == 0
            obj, opp, cut = max if is_player else min, min if is_player else max, alpha_cut if is_player else beta_cut
            return estimate(state.generateSuccessor(agent, action), next_agent, obj, opp, depth, alpha, beta, cut)[0], action

        def estimate(state, agent, objective, opposite, depth, alpha, beta, cut_f):
            if state.isLose() or state.isWin() or depth <= 0:
                return self.evaluationFunction(state), None

            next_agent, depth = (0, depth - 1) if (agent + 1) == state.getNumAgents() else (agent + 1, depth)
            next_state = (opposite(float('inf'), float('-inf')), Directions.STOP)

            for action in state.getLegalActions(agent):
                candidate = successor(state, agent, next_agent, depth, alpha, beta, action)
                next_state = candidate if objective(candidate[0], next_state[0]) == candidate[0] and candidate[0] != next_state[0] else next_state
                alpha, beta, is_cut = cut_f(alpha, beta, next_state[0])
                if is_cut:
                    break
            return next_state

        return estimate(gameState, 0, max, min, self.depth, float("-inf"), float("inf"), alpha_cut)[1]
        # END_YOUR_ANSWER

######################################################################################
# Problem 3a: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (problem 3)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """

        # BEGIN_YOUR_ANSWER (our solution is 30 lines of code, but don't worry if you deviate from this)
        def successor(state, agent, next_agent, depth, action):
            is_player = next_agent == 0
            obj, opp = max if is_player else min, min if is_player else max
            return estimate(state.generateSuccessor(agent, action), next_agent, obj, opp, depth)[0], action

        def estimate(state, agent, objective, opposite, depth):
            if state.isLose() or state.isWin() or depth <= 0:
                return self.evaluationFunction(state), None

            accumulator, actions = 0, state.getLegalActions(agent)
            next_agent, depth = (0, depth - 1) if (agent + 1) == state.getNumAgents() else (agent + 1, depth)
            next_state = (opposite(float('inf'), float('-inf')), Directions.STOP)

            for action in actions:
                candidate = successor(state, agent, next_agent, depth, action)
                accumulator += candidate[0]
                next_state = candidate if objective(candidate[0], next_state[0]) == candidate[0] and candidate[0] != next_state[0] else next_state

            return next_state if agent == 0 else (accumulator / len(actions), None)

        return estimate(gameState, 0, max, min, self.depth)[1]
        # END_YOUR_ANSWER

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
    """
    Your extreme, unstoppable evaluation function (problem 4).
    """

    # BEGIN_YOUR_ANSWER (our solution is 60 lines of code, but don't worry if you deviate from this)
    n_agents = currentGameState.getNumAgents()

    p_capsules = currentGameState.getCapsules()
    p_pacman = currentGameState.getPacmanPosition()
    p_ghosts = currentGameState.getGhostPositions()

    s_ghosts = currentGameState.getGhostStates()

    g_food = currentGameState.getFood().asList()
    g_walls = currentGameState.getWalls()

    scared_ghosts = [g for g in s_ghosts if g.scaredTimer]
    normal_ghosts = [g for g in s_ghosts if not g.scaredTimer]

    food_distances = [manhattanDistance(p_pacman, food) for food in g_food] + [0]
    capsule_distances = [manhattanDistance(p_pacman, capsule) for capsule in p_capsules] + [0]
    scared_ghost_distances = [manhattanDistance(p_pacman, ghost.getPosition()) for ghost in scared_ghosts] + [0]
    normal_ghost_distances = [manhattanDistance(p_pacman, ghost.getPosition()) for ghost in normal_ghosts] + [0]

    sum_food = sum(food_distances)
    fit_food = min(food_distances)
    num_food = len(food_distances) - 1

    sum_capsule = sum(capsule_distances)
    fit_capsule = min(capsule_distances)
    num_capsule = len(capsule_distances) - 1

    sum_scared = sum(scared_ghost_distances)
    fit_scared = min(scared_ghost_distances)

    sum_normal = sum(normal_ghost_distances)
    fit_normal = min(normal_ghost_distances)

    weight = [1.5, 4, 0, 20, 2, 2]
    phi = [fit_food, num_food, fit_capsule, num_capsule, fit_scared, (1.0 / len(s_ghosts))]

    return scoreEvaluationFunction(currentGameState) - sum([p * w for p in phi for w in weight])
    # END_YOUR_ANSWER

# Abbreviation
better = betterEvaluationFunction

