"""
exploring the state space with on policy MCTS, whci means that
the behavior policy (default policy =  the policy used to perform rollouts) is also the policy that is gradually refined via learning (the target policy)

In this project, a neural network (ANET) constitutes the target policy. It takes a board
state as input and produces a probability distribution over all possible moves (from that state) as output
"""
import math
import random
import pickle
import numpy as np
from random import randint

from keras.layers import Dense, Dropout
from keras.regularizers import l2

from keras import initializers, optimizers
from keras.models import Model, Sequential, load_model
from typing import List, Tuple
from hexgame import HexGameState
import keras.backend as K
from keras.optimizers import Adam


def random_target_policy(state: Tuple[int]|HexGameState, actions: List[Tuple[int] | int]):
    """
    children are to be the children of the node
    """
    # TODO: to be dependent on the player, player 2 wants to maximize score, player 2 wants to minimize score
    return actions[randint(0, len(actions)-1)]

# player 1 (black) wants to maximize

class ANET():

    def __init__(self, board_size, model=None, eps=0.2,  method="most-probable") -> None:
        self.eps = eps
        self.method = method
        self.board_size = board_size
        if model is None:
            #self.model = Sequential()
            #self.model.add(layers.Input(shape=(input_size,))) # 1 neuron for player, rest for board
            #self.model.add(layers.Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            #self.model.add(layers.Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), bias_initializer=initializers.Zeros()))
            #self.model.add(layers.Dense(units=16, activation='softmax'))
            #self.model.compile(loss='categorical_crossentropy',  optimizer='adam'), 
            model = Sequential()
            model.add(Dense(64, input_dim=board_size**2+1, activation='relu')) #, kernel_regularizer=l2(0.001)
            model.add(Dropout(0.2))
            model.add(Dense(32, activation='relu')) #, kernel_regularizer=l2(0.001)
            model.add(Dropout(0.1))
            model.add(Dense(board_size**2, activation='softmax'))
            optimizer = Adam(learning_rate=0.0005)
            model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
            self.model = model
        else:
            self.model = model

    def fit(self, X, y, epochs=50, batch_size=16, validation_data=(None, None)):
        x_valid, y_valid = validation_data
        if x_valid is not None and y_valid is not None:
            valid_loss = K.mean(K.categorical_crossentropy(y_valid, self.model.predict(x_valid)))
            print(f"Validation loss is {valid_loss}")
        self.model.fit(X, y, model__epochs=epochs, model__batch_size=batch_size)  #TODO : generalize

    def predict(self, state_1D): 
        output = self.model.predict(state_1D.reshape((1,-1)), verbose=0)
        return output


    def target_policy(self, state:HexGameState, actions: List[Tuple[int]]|None=None, method = None):
        """
        With probability ε, a random move is taken; 
        and with a probability of 1−ε, the move corresponding to the highest value in D is chosen."
        """
        eps = self.eps
        probs = self.predict(state.to_1D()) 
        action_anet_outputs = [action[0]*self.board_size+action[1] for action in actions]
        mask = [1 if i in action_anet_outputs else 0 for i in range(self.board_size**2)]
        probs = probs *mask
        probs /= np.sum(probs) 
        
        if random.random() < eps:
            action =  random.choice(actions)# random action from the list of all actions
            return action

        if method is None:
            method = self.method
        if method == "use-distribution":
            action_index = np.random.choice(range(self.board_size**2), p=probs[0])

        elif method == "most-probable":
            action_index = np.argmax(probs)

        else:
            raise Exception(f"There is no method named {method}")
        
        action = action_index//self.board_size, action_index%self.board_size #TODO: generalize

        return action
    
    def save(self, path, is_pipeline=False):
        if is_pipeline:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            self.model.save(path)

    @staticmethod
    def load(path, is_pipeline=False):
        if is_pipeline:
            with open(path, 'rb') as f:
                pipeline = pickle.load(f)
                board_size = pipeline.named_steps['preprocess'].kw_args['board_size']
                return ANET(board_size=board_size, model=pipeline)
        model = load_model(path)
        input_shape = model.input_shape
        board_size = int(math.sqrt(input_shape[1]-1))
        anet = ANET(board_size=board_size, model=model)
        return anet

