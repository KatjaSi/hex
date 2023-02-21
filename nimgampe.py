from enum import Enum
from game import Game
from typing import Tuple


class NimGame(Game):
    """
    The Game is to take an amount of pices, min 1, max K. The player who takes the last piece wins
    """
    def __init__(self, *args, **kwargs):
        if kwargs: # if initialized by named parameters
            self.N = kwargs.get('N')       
            self.K = kwargs.get('K') 
            self.player_to_move = kwargs.get('player_to_move') 
        elif args: # if initialized by game state, which is in this case a tuple (player, N, K). Player is 1 or 2
            self.player_to_move = args[0]
            self.N = args[1]
            self.K = args[2]
        else:
            raise Exception("The initialization can not be done by the specified parameters or no parameters were given")
        


    def make_move(self, n):
        """
        The player removes n stones from the board
        """
        if n < 1 or n > self.K or n > self.N:
            raise Exception("The move is not legal")
        self.N -= n
        self.player_to_move == self.player_to_move % 2 + 1


    def is_game_finished(self):
        return self.N == 0

    def is_player_won(self, player):
        return self.is_game_finished() and self.player_to_move != player

    def generate_successor_game_states(self):
        game_states = list()
        for i in range(1, min(self.K+1, self.N+1)):
            game_states.append(( self.player_to_move % 2 + 1, self.N-i, self.K))
        return game_states

    def get_game_state(self) -> Tuple[int]:
        """
        returns tuple (x,y,z) where x is player to move, y is number of pieces on the board, K is the max number of peces to take at once
        """
        return (self.player_to_move, self.N, self.K)

    def __str__(self) -> str:
        return f"N = {self.N} K = {self.K} player to move is {self.player_to_move.value}"

    def __repr__(self):
       return self.__str__()


def main() -> None:
    game = NimGame(*(1, 10, 2))
    game.make_move(2)
    game.make_move(1)
    game.make_move(2)
    #suc_gs = game.generate_successor_game_states()
    print(game.get_game_state())


if __name__ == '__main__':
    main()

