# Import and initialize your own actor 
from policy.target_policy import ANET
from hexgame import HexGame, HexStateManager, HexGameState
import numpy as np

anet = ANET.load("anets7x7/anet500.h5", board_size=7)
anet.eps = 0.0
# Import and override the `handle_get_action` hook in ActorClient
from ActorClient import ActorClient
class MyClient(ActorClient):
 def handle_get_action(self, state):
    board = np.array(state[1:]).reshape((7,7))
    state = HexGameState(player=state[0], board=board, red_unions=None, black_unions=None)
    legal_moves = HexStateManager.get_all_legal_actions(state=state)
    row, col =  anet.target_policy(state=state, actions=legal_moves) #TODO. change state
    return int(row), int(col)
 
# Initialize and run your overridden client when the script is executed
if __name__ == '__main__':
 client = MyClient()
 client.run()