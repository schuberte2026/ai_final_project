import numpy as np
from collections import defaultdict

#@@@@@@@@@@@@@@@@@@@@@@@@@@@
#a class for the environment
#@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Agent:
    '''A class to manage the agent'''
    
    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def __init__(self, actions, alpha=0.001, gamma=0.9, epsilon=0.1, epsilon_min=0.001, epsilon_max=0.99, epsilon_decay=0.999, do_epsilon_decay=False):
        '''Set up the constructor
            Takes -- config, a dictionary specifying the track dimensions and initial state
        '''
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_decay = epsilon_decay
        self.do_epsilon_decay = do_epsilon_decay
        self.actions = actions
        self.Q = defaultdict(lambda: defaultdict(lambda: 0.0))


    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def pi(self, s):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.actions)
        
        q_vals = self.Q[s]
        return max(q_vals, key=q_vals.get) if q_vals else np.random.choice(self.actions)

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def update_Q_learning(self,new_state,action_taken,reward,previous_state, done):
        # self.q_table[s_pr, a] += self.alpha * \
        # (r_pr + self.gamma * np.max(self.q_table[s]) - \
        #   self.q_table[s_pr, a])
        
        q_state_action = self.Q[previous_state][action_taken]
        if done:
            target = reward
        else:
            next_q_values = self.Q[new_state].values()
            target = reward + self.gamma * (max(next_q_values) if next_q_values else 0.0)

        # update
        self.Q[previous_state][action_taken] = q_state_action + \
                                       self.alpha * (target - q_state_action)
        if self.do_epsilon_decay:
            self.decay_epsilon()

    #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)