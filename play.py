from hexgame import HexGame, HexStateManager
from gui import Hex
from controller import Controller, ANET_Controller
from mcts import MCTS
from policy.tree_policy import max_tree_policy, min_tree_policy
from policy.target_policy import random_target_policy, ANET

game = HexGame(4)
game_view = Hex(size=4)

from policy.target_policy import ANET

anet = ANET.load("anet.h5")

#s_init = HexStateManager.generate_initial_state(size=4)
#mcts = MCTS(SM=HexStateManager, state=s_init, tree_policy=(max_tree_policy, min_tree_policy), target_policy=anet.target_policy, M=6)
#controller = Controller(game_view=game_view, game_model=game, mcts=mcts)
#controller.start_game()

anet_controller = ANET_Controller(game_view=game_view, game_model=game, anet=anet)
anet_controller.start_game()