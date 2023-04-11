"""
RL algorithm to get the target policy that will be used in the tournament later
"""
import numpy as np
from policy.target_policy import ANET
from hexgame import HexStateManager, HexGame
from mcts import MCTS
from policy.tree_policy import max_tree_policy, min_tree_policy
import tensorflow as tf
from pipelines import build_conv_pipeline, build_seq_pipeline

class RBUF:
    """
    class for preparing and storing the training data for ANET.
    """
    def __init__(self, board_size) -> None:
        self.X = list()
        self.y = list()
        self.board_size = board_size

    def clear(self):
        self.X = list()
        self.y = list()

    def add_case(self, state_1D, all_moves, D):
        """
        transforms the input information into the distribution over all the moves and then
        adds training case to the buffer
        """
       # if len(self.X) > 500:
        #    self.X = self.X[250:]
         #   self.y = self.y[250:]
        max_val = np.max(list(D.values()))
        for k in D:
            D[k] = D[k]/ max_val
        x = state_1D 
        all_y_valls = np.array(list(D.values()))
        #all_y_valls = all_y_valls-max(all_y_valls) # normalize so not to overflow exp
        soft_sum = np.sum(np.exp(all_y_valls))  #TODO: overloading is here! check and fix!
        y = np.array([np.e**D[move]/soft_sum if move in D  else 0 for move in all_moves])

         # check if the state already exists in the buffer
        if len(self.X)>0 and (x == self.X).all(-1).any(): 
            idx = np.where(np.all(self.X == x, axis=1))[0][0]
            self.y[idx] = y
        else:
            self.X.append(x)
            self.y.append(y)
        print(f"Number of unique elements in buffer is {len(np.unique(self.X, axis=0))}\nnumber of elements is {len(self.X)}")

    def get_training_data(self):
        return np.array(self.X), np.array(self.y,dtype='float32')
        
    def move_to_input(self, move):
        """
        Connects the move e.g. (1,2) to the corresponding index in ANET output layer, e.g. 
        """
        i,j = move
        return self.board_size*i+j
    
    def save(self, path):
        np.save(f'{path}/X.npy', self.X)    
        np.save(f'{path}/y.npy', self.y) 

    @classmethod #TODO: check
    def load(cls, path, board_size):
        rbuf = cls(board_size)
        rbuf.X = [np.array(x) for x in np.load(f'{path}/X.npy', allow_pickle=True)]
        rbuf.y = [np.array(y) for y in np.load(f'{path}/y.npy', allow_pickle=True)]
        return rbuf
        

def run_RL_algorithm(g_a, anet:ANET, rbuf:RBUF, interval:int):
    #rbuf.clear()
    board_size = anet.board_size
    for i in range(19,g_a):
        player = i % 2 +1
        game = HexGame(board_size, player=player)
        state = HexStateManager.generate_initial_state(size=board_size, player=player) # TODO:generalize
        mcts = MCTS(SM=HexStateManager, state=state, tree_policy=(max_tree_policy, min_tree_policy), target_policy=anet.target_policy, M=300)
        while not game.is_game_finished():
            state = mcts.root.state
            mcts.simulate()
            # D = distribution of visit counts in MCT along all arcs emanating from root
            all_moves  = HexStateManager.get_all_actions(state)
            legal_moves_with_visits = mcts.get_visits_distributions()
            rbuf.add_case(state_1D=state.to_1D(), all_moves=all_moves, D = legal_moves_with_visits) #
            move = mcts.get_move()
            mcts.reset_root(move)
            game.make_move(move)
        # Train ANET on a random minibatch of cases from RBUF
        X, y = rbuf.get_training_data()
        batch_size = min(len(X), 48) 
        indices = np.random.choice(len(X), batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]

        valid_data =  X[-8:], y[-8:] # the last added data
        anet.fit(X_batch, y_batch, epochs=40, validation_data=valid_data)
        #anet.fit(*rbuf.get_training_data(), epochs=50)
        mcts.target_policy = anet.target_policy #this line added
        print(i)
        rbuf.save("rbuf4x4_2xseq")
        if i%interval == 0:
            #anet.save(f"anets3x3conv/anet{i}.h5")
            anet.save(f"anets_4x4_2xseq/anet{i}.h5", is_pipeline=True)
            
    
    anet.save("anets/anet_final.h5")

#conv_pipeline = build_conv_pipeline(board_size=4)
seq_pipeline = build_seq_pipeline(board_size=4)
#anet = ANET(board_size=4, method="use-distribution", model = seq_pipeline) #1 +7*7
anet = ANET.load("anets_4x4_2xseq/anet18.h5", is_pipeline=True, board_size=4)
anet.method = "use-distribution" 
anet.eps = 0.05
#rbuf = RBUF(4)
rbuf = RBUF.load('rbuf4x4_2xseq', board_size=4)
run_RL_algorithm(201,anet, rbuf, interval=1)