"""
Module for game board and gam cell representation
"""
from copy import deepcopy
from typing import List, Tuple
from statemanager import StateManager
import numpy as np
from itertools import chain


from statemanager import StateManager

class HexGameState:
    
    def __init__(self, player, board, black_unions, red_unions) -> None:
        self.player = player
        self.board = board
        self.black_unions = black_unions
        self.red_unions = red_unions
    
    def to_1D(self):
        return  np.array(list(chain.from_iterable([[self.player]]+[list(chain.from_iterable(self.board))])))

    def __str__(self):
        return f"player: {self.player}\nboard: {self.board}\nblack unions: {self.black_unions}\nred unions: {self.red_unions}"
    

 
class HexStateManager:

    NEIGHBOUR_INDICES = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]

    @staticmethod
    def generate_initial_state(size, player) -> HexGameState:
        board = [[0 for _ in range(size)] for _ in range(size)]
        return HexGameState(player=player, board=board, black_unions=list(), red_unions=list())
    
    @staticmethod
    def generate_child_states(state: HexGameState, limit: int|None = None) -> List[HexGameState]:
        current_board_state = deepcopy(state.board)
        game_states = list()
        size = len(current_board_state)
        empty_cells= [(i,j)for i in range(size) for j in range(size) if current_board_state[i][j]==0]
        for pos in empty_cells:
            hgs = HexStateManager.generate_child_state(state=state, action=pos)
            game_states.append(hgs)   
        return game_states

    @classmethod
    def generate_child_state(cls, state: HexGameState, action: Tuple[int,int]) -> HexGameState:
        pos = action
        player = state.player
        board = deepcopy(state.board)
        if board[pos[0]][pos[1]]>0:
            raise Exception("The action is not legal")
        else:
            board[pos[0]][pos[1]] = player
        unions = None
        if player == 1:
            unions = deepcopy(state.black_unions)
        else:
            unions = deepcopy(state.red_unions)
        sets_to_join = list()
        size = len(board)
        for neighbour in cls.__get_neighbours__(pos, size):
            if board[neighbour[0]][neighbour[1]] == player:
                for s in unions: # find the union that has neigbour cell
                    if s not in sets_to_join and neighbour in s:
                        s.add(pos)
                        sets_to_join.append(s)
                        break
        if len(sets_to_join) == 0:
            s = set()
            s.add(pos)
            unions.append(s) 
        if len(sets_to_join) > 1:
            union_of_sets = set().union(*sets_to_join)
            unions =  [el for el in unions if el not in sets_to_join]
            unions.append(union_of_sets)
        next_player = player % 2 +1
        if player == 1: 
            return HexGameState(player=next_player, board=board, black_unions=unions, red_unions=deepcopy(state.red_unions))
        return HexGameState(player=next_player, board=board, black_unions=deepcopy(state.black_unions), red_unions=unions) 
    
    @classmethod
    def is_game_finished(cls, state: HexGameState) -> bool:
        return cls.black_is_won(state) or cls.red_is_won(state)
    
    @staticmethod
    def get_all_legal_actions(state: HexGameState) -> List[Tuple[int]]:
        board = state.board
        size=(len(board))
        return [(i,j)for i in range(size) for j in range(size) if board[i][j]==0]
    
    @staticmethod
    def get_all_actions(state: HexGameState) -> List[Tuple[int]]:
        """
        Returns the list of all actions, both legal and illegal in the form (0,0), (0,1), ... (n-1,n-1) n is the size of game board
        """
        board = state.board
        size=(len(board))
        return [(i,j)for i in range(size) for j in range(size)]
    
    @staticmethod
    def get_player(state: HexGameState) -> int:
        return state.player

    @classmethod
    def black_is_won(cls, state: HexGameState):
        # black wants to connect NW && SE
        size = len(state.board)
        for u in state.black_unions:
            nw = 0
            se = 0
            for cell in u:
                if cls.__is_cell_NW__(cell, size):
                    nw +=1
                elif cls.__is_cell_SE__(cell, size):
                    se +=1
            if nw > 0 and se > 0:
                return True
        return False
    
    @classmethod
    def red_is_won(cls, state: HexGameState):
        # black wants to connect NE && SW
        size = len(state.board)
        for u in state.red_unions:
            ne = 0
            sw = 0
            for cell in u:
                if cls.__is_cell_NE__(cell, size):
                    ne +=1
                elif cls.__is_cell_SW__(cell, size):
                    sw +=1
            if ne > 0 and sw > 0:
                return True
        return False

    @staticmethod   
    def __is_cell_NW__(pos, size=None):
        return pos[0] == 0

    @staticmethod  
    def __is_cell_NE__(pos, size):
        return pos[1] == size-1

    @staticmethod  
    def __is_cell_SE__(pos, size):
        return pos[0] == size-1

    @staticmethod  
    def __is_cell_SW__(pos, size=None):
        return pos[1] == 0

    @staticmethod     
    def __transpose__(board):
        size = len(board)
        return [[board[j][i] for j in range(size)] for i in range(size)]
    
    @classmethod  
    def __get_neighbours__(cls, pos:Tuple[int, int], size, condition: bool|None = None) -> List[Tuple[int,int]]:
        i, j = pos
        neighbours = list()
        for ind in cls.NEIGHBOUR_INDICES:
            if (i+ind[0] >= 0 and i+ind[0] < size  and j+ind[1] >= 0 and j+ind[1] < size):
                neighbours.append((i+ind[0], j+ind[1]))
        return neighbours
    
    @staticmethod
    def __intersection__(lst1, lst2):
        return list(set(lst1) & set(lst2))

    
    @staticmethod
    def get_winner(state: Tuple):
        return 1 if HexStateManager.black_is_won(state) else 2

class HexGame():

    def __init__(self, size, player):
        self.state = HexStateManager.generate_initial_state(size=size, player=player)
        self.size = len(self.state.board)
        self.state_manager = HexStateManager

    def make_move(self, pos):
        self.state = self.state_manager.generate_child_state(state=self.state, action=pos)

    def black_is_won(self):
        # black wants to connect NW && SE
        return self.state_manager.black_is_won(self.state)

    def red_is_won(self):
        return self.state_manager.red_is_won(self.state)
    
    def get_player_to_move(self):
        return self.state.player

    def get_legal_moves(self):
        return HexStateManager.get_all_legal_actions(self.state)

    def is_game_finished(self):
        return self.red_is_won() or self.black_is_won()

    def get_winner(self):
        return 1 if self.black_is_won() else 2
