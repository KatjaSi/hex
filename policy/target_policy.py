"""
exploring the state space with on policy MCTS, whci means that
the behavior policy (default policy =  the policy used to perform rollouts) is also the policy that is gradually refined via learning (the target policy)

In this project, a neural network (ANET) constitutes the target policy. It takes a board
state as input and produces a probability distribution over all possible moves (from that state) as output
"""
import random
import numpy as np
from random import randint

from keras.layers import Dense, Dropout
from keras.regularizers import l2

from keras import initializers, optimizers
from keras.models import Model, Sequential, load_model
from typing import List, Tuple
from hexgame import HexGameState
import keras.backend as K

def random_target_policy(state: Tuple[int]|HexGameState, actions: List[Tuple[int] | int]):
    """
    children are to be the children of the node
    """
    # TODO: to be dependent on the player, player 2 wants to maximize score, player 2 wants to minimize score
    return actions[randint(0, len(actions)-1)]

# player 1 (black) wants to maximize

class ANET():

    def __init__(self, model=None, input_size=17, eps=0.2,  method="most-probable") -> None:
        self.eps = eps
        self.method = method
        if model is None:
            #self.model = Sequential()
            #self.model.add(layers.Input(shape=(input_size,))) # 1 neuron for player, rest for board
            #self.model.add(layers.Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            #self.model.add(layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            #self.model.add(layers.Dense(units=16, activation='softmax'))
            #self.model.compile(loss='categorical_crossentropy',  optimizer='adam'), 
            model = Sequential()
            model.add(Dense(64, input_dim=17, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.5))
            model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
            model.add(Dropout(0.5))
            model.add(Dense(16, activation='softmax'))
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        else:
            self.model = model

    def fit(self, X, y, epochs=10, batch_size=32):
        x_valid = X[-10:] # 10 last
        y_valid = y[-10:] # 10 last elements, they will be the new elements , most recently added
        valid_loss = K.mean(K.categorical_crossentropy(y_valid, self.model.predict(x_valid)))
        print(f"Validation loss is {valid_loss}")
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, state_1D): 
        output = self.model.predict(state_1D.reshape((1,-1)), verbose=0)
        return output


    def target_policy(self, state:HexGameState, actions: List[Tuple[int]]|None=None, method = None):
        """
        With probability ε, a random move is taken; 
        and with a probability of 1−ε, the move corresponding to the highest value in D is chosen."
        """
        eps = self.eps
        probs = self.predict(state.to_1D()) # TODO: ?
        action_anet_outputs = [action[0]*4+action[1] for action in actions]
        mask = [1 if i in action_anet_outputs else 0 for i in range(16)]
        probs = probs *mask
        probs /= np.sum(probs) 
        
        if random.random() < eps:
            action =  random.choice(actions)# random action from the list of all actions
            return action

        if method is None:
            method = self.method
        if method == "use-distribution":
            action_index = np.random.choice(range(16), p=probs[0])

        elif method == "most-probable":
            action_index = np.argmax(probs)

        else:
            raise Exception(f"There is no method named {method}")
        
        action = action_index//4, action_index%4 #TODO: generalize

        return action
    
    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        model = load_model(path)
        anet = ANET(model)
        return anet

