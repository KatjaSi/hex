from enum import Enum
from typing import Tuple, List
from statemanager import StateManager
import math

class NimStateManager(StateManager):

    def __init__(self, state) -> None:
        self.state = state

    @staticmethod
    def generate_initial_state(N,K) -> Tuple:
        return (1,N,K)

    @staticmethod
    def generate_child_states(state: Tuple, **kwargs) -> List[Tuple]:
        game_states = list()
        player = state[0]
        N = state[1]
        K = state[2]
        limit = kwargs.get("limit") if "limit" in kwargs and kwargs.get("limit") is not None else math.inf
        for i in range(1, min(K+1, N+1, limit+1)):
            game_states.append(( player % 2 + 1, N-i, K))
        return game_states

    @staticmethod
    def generate_child_state(state: Tuple, action: int) -> Tuple:
        player, N, K = state
        if action < 1 or action > K or action > N:
            raise Exception("The move is not legal")
        N -= action
        player = player % 2 + 1
        return (player, N, K)
    
    @staticmethod
    def get_all_actions(state: Tuple) -> List[int]:
        _, N, K = state
        actions = list()
        for i in range(1, min(N+1,K+1)):
            actions.append(i)
        return actions
    
    @staticmethod
    def get_player(state: Tuple) -> int:
        return state[0]
 

    @staticmethod
    def is_game_finished(state: Tuple) -> bool:
        return state[1] == 0
    
    @staticmethod
    def get_winner(state: Tuple):
        player_to_move = state[0]
        return player_to_move % 2 + 1 if state[1] == 0 else None

class NimGame:
    """
    The Game is to take an amount of pieces, min 1, max K. The player who takes the last piece wins
    """
    def __init__(self, N, K):
        self.nsm = NimStateManager((N,K))
        self.state = self.nsm.generate_initial_state(N,K)       
            
    def make_move(self, n): 
        """
        The player removes n stones from the board
        """
        self.state = self.nsm.generate_child_state(state=self.state, action=n)

    def is_game_finished(self):
        return self.nsm.is_game_finished(self.state)
    
    def get_winner(self):
        return self.nsm.get_winner(state=self.state)

    def is_player_won(self, player):
        return self.nsm.get_winner(state=self.state) == player

    def __str__(self) -> str:
        player_to_move, N, K = self.state
        return f"N = {N} K = {K} player to move is {player_to_move.value}"

    def __repr__(self):
       return self.__str__()


def main() -> None:
    game = NimGame(10,2)
    print(game.state)
    game.make_move(2)
    print(game.state)
    game.make_move(1)
    print(game.state)
    game.make_move(2)
    print(game.state)
    game.make_move(1)
    print(game.state)
    game.make_move(2)
    print(game.state)
    game.make_move(2)
    print(game.state)
    b = game.is_player_won(2)
    print(b)
    nsm = NimStateManager((10,2))
    state0 = nsm.generate_initial_state(N=6, K=2)
    print(nsm.generate_child_states(state=state0, limit = None))
    #state1 = nsm.generate_child_state(state=state0, action=2)
    #print(state1)
    #state2 = nsm.generate_child_state(state=state1, action=2)
    #print(state2)
    #state3 = nsm.generate_child_state(state=state2, action=2)
    #print(state3)
    #print(nsm.get_winner(state=state3))


if __name__ == '__main__':
    main()

