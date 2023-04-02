"""
exploring the state space with on policy MCTS, whci means that
the behavior policy (default policy =  the policy used to perform rollouts) is also the policy that is gradually refined via learning (the target policy)

In this project, a neural network (ANET) constitutes the target policy. It takes a board
state as input and produces a probability distribution over all possible moves (from that state) as output
"""
import numpy as np
from random import randint
from keras import layers
from keras import initializers, optimizers
from keras.models import Model, Sequential, load_model
from typing import List, Tuple
from hexgame import HexGameState


def random_target_policy(state: Tuple[int]|HexGameState, actions: List[Tuple[int] | int]):
    """
    children are to be the children of the node
    """
    # TODO: to be dependent on the player, player 2 wants to maximize score, player 2 wants to minimize score
    return actions[randint(0, len(actions)-1)]

# player 1 (black) wants to maximize

class ANET():

    def __init__(self, model=None, input_size=17) -> None:
        if model is None:
            self.model = Sequential()
            self.model.add(layers.Input(shape=(input_size,))) # 1 neuron for player, rest for board
            self.model.add(layers.Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            self.model.add(layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            self.model.add(layers.Dense(units=16, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy')
        else:
            self.model = model

    def fit(self, X, y, epochs=10, batch_size=32):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, state_1D): # [sum([[player]] + board1D,[])]
        output = self.model.predict(state_1D.reshape((1,-1)), verbose=0)
        return output


    def target_policy(self, state:HexGameState, actions: List[Tuple[int]]|None=None):
        probs = self.predict(state.to_1D()) # TODO: ?
        #probs *= mask  # Zero-out the probabilities of illegal moves
        #probs /= np.sum(probs) 
        action_anet_outputs = [action[0]*4+action[1] for action in actions]
        mask = [1 if i in action_anet_outputs else 0 for i in range(16)]
        probs = probs *mask
        probs /= np.sum(probs) 
        max_index = np.argmax(probs)
       
        action = max_index//4, max_index%4 #TODO: generalize

        return action
    
    def save(self, path):
        self.model.save(path)

    @staticmethod
    def load(path):
        model = load_model(path)
        anet = ANET(model)
        return anet

