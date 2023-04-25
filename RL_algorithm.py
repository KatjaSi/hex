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
from rbuf import RBUF

def run_RL_algorithm(g_a, anet:ANET, rbuf:RBUF, interval:int):
    #rbuf.clear()
    board_size = anet.board_size
    for i in range(g_a):
        player = i % 2 +1
        anet.eps -= i/200
        game = HexGame(board_size, player=player)
        state = HexStateManager.generate_initial_state(size=board_size, player=player) 
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
       # batch_size = min(len(X), 64) 
        batch_size = int(len(X)) 
        indices = np.random.choice(len(X), batch_size, replace=False)
        X_batch, y_batch = X[indices], y[indices]

        valid_data =  X[-10:], y[-10:] # the last added data
        anet.fit(X_batch, y_batch, epochs=40, validation_data=valid_data)
        #anet.fit(*rbuf.get_training_data(), epochs=50)
        mcts.target_policy = anet.target_policy #this line added
        #print(i)
        rbuf.save("rbuf4x4")
        if i%interval == 0:
            anet.save(f"anets_lm/anet{i}.h5")
            

#conv_pipeline = build_conv_pipeline(board_size=4)
#seq_pipeline = build_seq_pipeline(board_size=4)
anet = ANET(board_size=5) #1 +7*7
#anet = ANET.load("anets_7x7/anet18.h5", is_pipeline=False, board_size=4)
#anet.method = "most-probable" 
anet.eps = 0.96
rbuf = RBUF(5)
#rbuf = RBUF.load('rbuf7x7', board_size=7)
run_RL_algorithm(201,anet, rbuf, interval=50)