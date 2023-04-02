import abc
from typing import List, Tuple

class StateManager(metaclass=abc.ABCMeta):
    """
    Interface for state management
    """

    @abc.abstractmethod
    def generate_initial_state(*args, **kwargs) -> Tuple:
        pass

    @abc.abstractmethod
    def generate_child_states(state: Tuple, *args, **kwargs) -> List[Tuple]:
        """
        Generates a list of child states, all child states, or possibly limited number of them. Depends on the implementation.
        """
        pass

    @abc.abstractmethod
    def generate_child_state(state: Tuple, action: object) -> Tuple:
        """
        Generates the child state, after applying the action to the given state
        """
        pass

    @abc.abstractmethod
    def is_game_finished(state: Tuple, action: object) -> bool:
        pass

    @abc.abstractmethod
    def get_winner(self, state: Tuple) -> int:
        pass
