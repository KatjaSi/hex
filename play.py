from hexgame import HexGame, HexStateManager
from gui import Hex
from controller import Controller, ANET_Controller
from mcts import MCTS
from policy.tree_policy import max_tree_policy, min_tree_policy
from policy.target_policy import random_target_policy, ANET

game = HexGame(4, player=2)
game_view = Hex(size=4)

from policy.target_policy import ANET

anet2 = ANET.load("anets_4x4/anet24.h5", is_pipeline=False)#ANET.load("anets/anet14.h5")
anet2.eps = 0
anet2.method = "use-distribution"
anet1 = ANET.load("anets_4x4/anet14.h5", is_pipeline=True)
anet1.eps = 0.0
anet1.method =  "most-probable"

#s_init = HexStateManager.generate_initial_state(size=4, player=2)
#mcts = MCTS(SM=HexStateManager, state=s_init, tree_policy=(max_tree_policy, min_tree_policy), target_policy=anet2.target_policy, M=300)
#controller = Controller(game_view=game_view, game_model=game, mcts=mcts)
#controller.start_game()

anet_controller = ANET_Controller(game_view=game_view, game_model=game, target_policy1 =anet1.target_policy, target_policy2=anet2.target_policy)
anet_controller.start_game()