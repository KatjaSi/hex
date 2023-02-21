import abc
from typing import List, Tuple

class Game(metaclass=abc.ABCMeta):
    """
    Interface for two-player games
    """
   
    @abc.abstractmethod
    def make_move(self) -> None:
        pass

    @abc.abstractmethod
    def is_game_finished(self) -> bool:
        pass
    
    @abc.abstractmethod
    def is_player_won(self, player) -> bool:
        pass

    @abc.abstractmethod    
    def generate_successor_game_states(self) -> List[Tuple[int]]:
        pass

    @abc.abstractmethod    
    def get_game_state(self) -> Tuple[int]:
        pass