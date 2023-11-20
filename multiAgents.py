import random
import statistics

import util
from game import Agent
from pacman import GameState


class ReflexAgent(Agent):

    def getAction(self, gameState: GameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"
        # print(legalMoves[chosenIndex])

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    def __init__(self, evalFn='betterEvaluationFunction', depth='3'):
        self.index = 0
        self.evaluationFunction = better
        self.depth = int(depth)
        self.BEST_ACTION = None


class MinimaxAgent(MultiAgentSearchAgent):

    def minimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = float("-inf")
            actions = gameState.getLegalActions(agent)
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                value_update = self.minimax(successor, depth, agent + 1)
                if depth == self.depth and value_update > value:
                    self.BEST_ACTION = action
                value = max(value, value_update)
            return value
        else:
            value = float('inf')
            n_agent = (agent + 1) % gameState.getNumAgents()
            if agent + 1 == gameState.getNumAgents():
                depth -= 1
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.minimax(successor, depth, n_agent))
            return value

    def getAction(self, gameState: GameState):
        self.minimax(gameState, self.depth, 0)
        return self.BEST_ACTION


class AlphaBetaAgent(MultiAgentSearchAgent):

    def minimax(self, gameState, depth, agent, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = float("-inf")
            actions = gameState.getLegalActions(agent)
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                value_update = self.minimax(successor, depth, agent + 1, alpha, beta)
                if depth == self.depth and value_update > value:
                    self.BEST_ACTION = action
                value = max(value, value_update)
                alpha = max(alpha, value)
                if value >= beta:
                    return value
            return value
        else:
            value = float('inf')
            n_agent = (agent + 1) % gameState.getNumAgents()
            if agent + 1 == gameState.getNumAgents():
                depth -= 1
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                value = min(value, self.minimax(successor, depth, n_agent, alpha, beta))
                beta = min(beta, value)
                if value <= alpha:
                    return value
            return value

    def getAction(self, gameState: GameState):
        self.minimax(gameState, self.depth, 0, float('-inf'), float('inf'))
        return self.BEST_ACTION


class ExpectimaxAgent(MultiAgentSearchAgent):

    def expectimax(self, gameState, depth, agent):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return self.evaluationFunction(gameState)

        if agent == 0:
            value = float("-inf")
            actions = gameState.getLegalActions(agent)
            for action in actions:
                successor = gameState.generateSuccessor(agent, action)
                value_update = self.expectimax(successor, depth, agent + 1)
                if depth == self.depth and value_update > value:
                    self.BEST_ACTION = action
                value = max(value, value_update)
            return value
        else:
            value = float('inf')
            scores = []
            n_agent = (agent + 1) % gameState.getNumAgents()
            if agent + 1 == gameState.getNumAgents():
                depth -= 1
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                list.append(scores, self.expectimax(successor, depth, n_agent))
                value = statistics.mean(scores)
            return value

    def getAction(self, gameState: GameState):
        self.expectimax(gameState, self.depth, 0)
        return self.BEST_ACTION


def nearest_food_distance(state):
    state.getFood()
    walls = state.getWalls()
    row, col = 0, 0
    for i in walls:
        for j in walls[0]:
            col += 1
        row += 1

    def is_in_bounds(i, j):
        if 0 < i < row and 0 < j < col:
            return True
        else:
            return False

    pac_position = state.getPacmanPosition()
    visited = set()
    queue = util.Queue()
    queue.push([pac_position, 0])
    while not queue.isEmpty():
        temp_position = queue.pop()
        x, y = temp_position[0]

        if state.hasFood(x, y):
            return temp_position[1]
        if temp_position[0] in visited:
            continue

        visited.add(temp_position[0])

        x, y = temp_position[0]
        if not walls[x - 1][y] and is_in_bounds(x - 1, y):
            queue.push([(x - 1, y), temp_position[1] + 1])
        if not walls[x + 1][y] and is_in_bounds(x + 1, y):
            queue.push([(x + 1, y), temp_position[1] + 1])
        if not walls[x][y - 1] and is_in_bounds(x, y - 1):
            queue.push([(x, y - 1), temp_position[1] + 1])
        if not walls[x][y + 1] and is_in_bounds(x, y + 1):
            queue.push([(x, y + 1), temp_position[1] + 1])

    return float('inf')


def betterEvaluationFunction(currentGameState: GameState):
    score = 0

    pac_pos = currentGameState.getPacmanPosition()
    food_remain = currentGameState.getNumFood()
    ghost_states = currentGameState.getGhostStates()
    ghost_distance = 0

    if currentGameState.isWin():
        return currentGameState.getScore() + 10000
    if currentGameState.isLose():
        return -10000

    score += currentGameState.getScore()
    score -= 100 * food_remain
    score -= nearest_food_distance(currentGameState)

    for ghost in ghost_states:
        d = util.manhattanDistance(ghost.getPosition(), pac_pos)
        ghost_distance += d
        if d < 3:
            score -= d * 100
    return score


# Abbreviation
better = betterEvaluationFunction
