from __future__ import annotations
from game import Game
from nimgampe import NimGame
from typing import Tuple

class Node:

    def __init__(self, state: Tuple[int], parent: Node | None = None )  -> None:
        self.state = state
        self.parent = parent
        self.children = list()


class MCTS:

    def __init__(self, game_cl: Game, state: Tuple[int]) -> None:
        self.root = Node(state=state)
        self.game_cl = game_cl

    def expand_node(self, node: Node, limit: int | None = None) -> None:
        game = self.game_cl(*node.state) 
        successor_states = game.generate_successor_game_states()
        for state in successor_states:
            child_node = Node(state=state, parent=node)
            node.children.append(child_node)


def main() -> None:
    mcts = MCTS(game_cl=NimGame, state=(1,10,2))
    mcts.expand_node(mcts.root)
     


if __name__ == '__main__':
    main()
    
    