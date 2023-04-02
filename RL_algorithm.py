"""
RL algorithm to get the target policy that will be used in the tournament later
"""
import numpy as np
from policy.target_policy import ANET
from hexgame import HexStateManager, HexGame
from mcts import MCTS
from policy.tree_policy import max_tree_policy, min_tree_policy


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
        x = state_1D 
        all_y_valls = np.array(list(D.values()))
        soft_sum = np.sum(np.exp(all_y_valls))
        y = np.array([np.e**D[move]/soft_sum if move in D  else 0 for move in all_moves])
        self.X.append(x)
        self.y.append(y)

    def get_training_data(self):
        return np.array(self.X), np.array(self.y)
        
    def move_to_input(self, move):
        """
        Connects the move e.g. (1,2) to the corresponding index in ANET output layer, e.g. 
        """
        i,j = move
        return self.board_size*i+j
        

def run_RL_algorithm(g_a, anet:ANET, rbuf:RBUF):
    rbuf.clear()
    for _ in range(g_a):
        game = HexGame(4)
        state = HexStateManager.generate_initial_state(size=4) # TODO:generalize
        mcts = MCTS(SM=HexStateManager, state=state, tree_policy=(max_tree_policy, min_tree_policy), target_policy=anet.target_policy, M=500)
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
            #print(sum(list(legal_moves_with_visits.values())))
        # train ANET
        anet.fit(*rbuf.get_training_data(), epochs=50)
        anet.save("anet.h5")

#state = HexStateManager.generate_initial_state(size=7)
#state1D = state.to_1D()
#print(state1D)
#prediction = anet.predict(state_1D=state1D)
#print(prediction)

#anet = ANET(input_size=17) #1 +7*7

anet = ANET.load("anet.h5")
rbuf = RBUF(4)
run_RL_algorithm(8,anet, rbuf)
