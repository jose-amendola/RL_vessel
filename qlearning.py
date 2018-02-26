import random
import math
import tilecoding

#This source indicates all possible actions
import actions

class QLearning:
    
    
    # Learning Rate
    alpha = None
    # Discount Rate
    gamma = None
    #episilon for exploration
    epsilon = None
    # Q-table
    qTable = None
    # Value to initiate Q-table
    initQ = None
    #Indicates if the agent is exploring the environment
    exploring = None

    #Set here if the agent uses TileCoding
    usesTile = True


    def __init__(self,alpha=0.1,epsilon=0.1,gamma=0.9,initQ = 0):
        """As parameters, please inform the learning rate (alpha), the discount factor (epsilon),
          and the value for initiating new Q-table entries (initQ)"""
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.qTable = {}
        self.initQ = initQ
        self.exploring = True

    
    def select_action(self, state):
        """ When this method is called, the agent executes an action based on its Q-table """
        #If exploring, an exploration strategy is executed
        if self.exploring:
            action =  self.exp_strategy(state)
        #Else the best action is selected according to the Q-table
        else:
            action = self.policy_check(state)
        
        return action

        
        
    def policy_check(self,state):
        """In case a fixed action is included in the policy cache, that action is returned
        else, the maxQ action is returned"""
        return self.max_Q_action(state)
        
        
    def max_Q_action(self,state):
        """Returns the action that corresponds to the highest Q-value"""
        actions = self.getPossibleActions()
        v,a =  self.get_max_Q_value_action(state,actions)
        return a

    def get_max_Q_value(self,state):
        """Returns the maximum Q value for a state"""
        actions = self.getPossibleActions()
        v,a =  self.get_max_Q_value_action(state,actions)
        return v
        
        
        
    def exp_strategy(self,state):
        """Returns the result of the exploration strategy"""

        allActions = self.getPossibleActions()
        prob = random.random()
        #epsilon-greeedy strategy
        if prob <= self.epsilon:
            return random.choice(allActions)
        return self.max_Q_action(state)
           

    
    def get_Q_size(self):
        """Returns the size of the QTable"""
        return len(self.qTable)
        
    
    def observe_reward(self,state,action,statePrime,reward, terminal_flag):
        """Performs the standard Q-Learning Update (only updated if the agent is exploring)"""
        if self.exploring:
            #TODO Include case for final states
            qValue= self.readQTable(state,action)
            V = self.get_max_Q_value(statePrime)
            if terminal_flag:
                newQ = qValue + self.alpha * (reward - qValue)
            else:
                newQ = qValue + self.alpha * (reward + self.gamma * V - qValue)
            # If the agent uses tile coding, the state is processed before accessing the Q-table
            if self.usesTile:
                state = tuple(tilecoding.tiling(state))
            self.qTable[(state,action)] = newQ

    def readQTable(self,state,action):             
        """Returns one value from the Qtable"""
        #If the agent uses tile coding, the state is processed before accessing the Q-table
        if self.usesTile:
            state = tuple(tilecoding.tiling(state))

        if not (state,action) in self.qTable:
            self.qTable[(state,action)] = self.initQ
        return self.qTable[(state,action)]

    def get_max_Q_value_action(self, state, allActions):
        """Returns the maximum Q value and correspondent action to a given state"""
        maxActions = []
        maxValue = -float('Inf')

        for act in allActions:
            # print str(type(state))+" - "+str(type(act))
            qV = self.readQTable(state, act)
            if (qV > maxValue):
                maxActions = [act]
                maxValue = qV
            elif (qV == maxValue):
                maxActions.append(act)

        #Chooses one of the best actions
        action = random.choice(maxActions)

        return maxValue, action

    def expose_QTable(self):
        return self.qTable

    def getPossibleActions(self):
        """Returns the possible actions"""
        
        return actions.all_agent_actions()

 