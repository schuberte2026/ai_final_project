import numpy as np
from collections import defaultdict

#@@@@@@@@@@@@@@@@@@@@@@@@@@@
#a class for the environment
#@@@@@@@@@@@@@@@@@@@@@@@@@@@
class VDBE_agent:
    '''A class to manage the agent'''
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, actions, alpha=0.001, gamma=0.9, epsilon_initial=1.0, sigma=0.1, delta=None):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the track dimensions and initial state
        '''
        # q-learning hyperparameters
        self.alpha = alpha
        self.gamma = gamma

        # vdbe hyperparameters
        self.sigma = sigma # inverse sensitivity parameter
        if delta is None: # influence of action on exploration rate update
            self.delta = 1 / len(actions) 
        self.epsilon_table = defaultdict(lambda: epsilon_initial) # state : epsilon mapping
        
        # environment parameters
        self.actions = actions
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def pi(self, s):
        epsilon = self.epsilon_table[s]
        if np.random.uniform() < epsilon:
            return np.random.choice(self.actions)
        
        q_vals = self.Q[s]
        return max(q_vals, key=q_vals.get) if len(q_vals) != 0 else np.random.choice(self.actions)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_Q_learning(self,new_state,action_taken,reward,previous_state, done):
        old_q = self.Q[previous_state][action_taken]
        if done:
            target = reward
        else:
            next_q_values = self.Q[new_state].values()
            target = reward + self.gamma * max(next_q_values) if len(next_q_values) != 0 else 0.0

        # update Q
        new_q = old_q + self.alpha * (target - old_q)
        self.Q[previous_state][action_taken] = new_q
        
        # update epsilon
        f = self.f(old_q, new_q) # VDBE
        old_epsilon = self.epsilon_table[previous_state] # get prev
        new_epsilon = self.update_epsilon(old_epsilon, f) # calculate new
        self.epsilon_table[previous_state] = new_epsilon # update!

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def f(self, q_old, q_new):
        """Compute the VDBE f-value based on change in Q."""
        diff = abs(q_new - q_old)
        x = diff / self.sigma
        return (1 - np.exp(-x)) / (1 + np.exp(-x)) # function value
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_epsilon(self, epsilon_old, f_value):
        """Update epsilon according to VDBE rule."""
        return self.delta*f_value + (1 - self.delta)*epsilon_old # new epsilon