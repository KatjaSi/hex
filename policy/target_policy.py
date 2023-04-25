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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from keras.layers import Dense, Dropout
from keras.regularizers import l2

from keras import initializers, optimizers
from keras.models import Sequential, load_model

from typing import List, Tuple
from hexgame import HexGameState
import keras.backend as K
from keras.optimizers import Adam
from torchkeras.lightmodel import LightModel
#from torchkeras.models import #TODO import smth instead of Sequential



def random_target_policy(state: Tuple[int]|HexGameState, actions: List[Tuple[int] | int]):
    """
    children are to be the children of the node
    """
    return actions[randint(0, len(actions)-1)]


class AnetModel(nn.Module):
    def __init__(self, board_size):
        super(AnetModel, self).__init__()
        self.layer1 = nn.Linear(board_size**2+1, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64,board_size**2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x
    
    def predict(self, inputs):
        self.eval() 
        with torch.no_grad():
            inputs = torch.tensor(inputs).float()
            outputs = self.forward(inputs)
            predictions = F.softmax(outputs, dim=-1)
        return predictions
    
    def fit(self, optimizer, criterion, train_loader, valid_data, epochs):
        print(valid_data[0][-1])
        print(valid_data[1][-1])
        print(self.predict(valid_data[0])[-1])
        for epoch in range(epochs):
            running_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            epoch_loss = running_loss 

            # Validation
            self.eval()
            inputs, labels = valid_data
            inputs, labels = torch.tensor(inputs).float(), torch.tensor(labels).float()
            outputs = self.forward(inputs)
            valid_loss = criterion(outputs, labels).item() 
            print('Epoch [{}/{}], Train Loss: {:.4f},  Valid Loss: {:.4f}'.format(epoch+1, epochs, loss.item(), valid_loss))

        
class ANET():

    def __init__(self, board_size, eps=0.2,  method="most-probable", model = None) -> None:
        self.eps = eps
        self.method = method
        self.board_size = board_size
        if model is None:
            self.model = AnetModel(board_size) 
        else:
            self.model = model
        self.criterion = nn.CrossEntropyLoss()   
        self.optimizer = torch.optim.Adam(self.model.parameters(),  lr=0.0002) #, weight_decay=0.0002
      #  self.light_model = LightModel(self.model)
     

    def fit(self, X, y, epochs=40, validation_data=(None, None)):            
        train_dataset = TensorDataset(torch.tensor(X).to(torch.float32), torch.tensor(y))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
        self.model.fit(optimizer=self.optimizer, criterion=self.criterion, train_loader=train_loader, valid_data=validation_data, epochs=epochs)
        #self.light_model = LightModel(self.model)

    def predict(self, state_1D): 
       # state_1d = state_1D.reshape((1,-1))
        return self.model.predict(state_1D)


    def target_policy(self, state:HexGameState, actions: List[Tuple[int]]|None=None, method = None):
        """
        With probability ε, a random move is taken; 
        and with a probability of 1−ε, the move corresponding to the highest value in D is chosen."
        """
        eps = self.eps
        probs = self.predict(state.to_1D()) 
        action_anet_outputs = [action[0]*self.board_size+action[1] for action in actions]
        mask = [1 if i in action_anet_outputs else 0 for i in range(self.board_size**2)]
        probs = probs.detach().numpy()*mask
        probs /= np.sum(probs)         
        if random.random() < eps:
            action =  random.choice(actions)# random action from the list of all actions
            return action
        if method is None:
            method = self.method
        if method == "use-distribution":
            action_index = np.random.choice(range(self.board_size**2), p=probs)
     
        elif method == "most-probable":
            action_index = np.argmax(probs)
        else:
            raise Exception(f"There is no method named {method}")
        
        action = action_index//self.board_size, action_index%self.board_size 
        return action
    
    def save(self, path, is_pipeline=False):
        if is_pipeline:
            with open(path, 'wb') as f:
                pickle.dump(self.model, f)
        else:
            #self.model.save(path)
            torch.save(self.model.state_dict(), path)

    @staticmethod
    def load(path, is_pipeline=False, board_size=5):
        if is_pipeline:
            with open(path, 'rb') as f:
                pipeline = pickle.load(f)
                #board_size = pipeline.named_steps['preprocess'].kw_args['board_size'] # for convolution
                return ANET(board_size=board_size, model=pipeline)
        saved_model_state_dict = torch.load(path)
        model = AnetModel(board_size)
        model.load_state_dict(saved_model_state_dict)
        anet = ANET(board_size=board_size, model=model)
        return anet

