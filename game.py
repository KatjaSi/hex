"""
Module for game board and gam cell representation
"""
from enum import Enum
from copy import deepcopy

class Player(Enum):
    BLACK = 1
    RED = 2
 

class Game:

    def __init__(self, size, board=None):
        self.size = size
        # board of cells where each cell knows its position on the board and position of its neighbours on the board
        if board is None:
            self.board = [[0 for j in range(size)] for i in range(size)]
            self.empty_cells= [(i,j)for i in range(size) for j in range(size)]
            self.player_to_move = Player.BLACK
        else:
            self.board = board
            self.empty_cells= [(i,j)for i in range(size) for j in range(size) if self.board[i][j]==0]
            player_to_move = (self.size**2 - len(self.empty_cells)) % 2 + 1
            if player_to_move == 1:
                self.player_to_move = Player.BLACK
            else:
                self.player_to_move = Player.RED

        neighbour_indices = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
        self.neighbours = dict()
        for i in range(size):
            for j in range(size):
                self.neighbours[(i, j)]=list()
                for ind in neighbour_indices:
                    if (i+ind[0] >= 0 and i+ind[0] < size  and j+ind[1] >= 0 and j+ind[1] < size):
                        self.neighbours[(i, j)].append((i+ind[0], j+ind[1]))

        self.players = [Player.BLACK, Player.RED] # black wants to connect NW && SE, red - SW && NE
        self.black_unions = list()
        self.red_unions = list()

    def is_cell_NW(self, pos):
        return pos[0] == 0

    def is_cell_NE(self, pos):
        return pos[1] == len(self.board)-1

    def is_cell_SE(self, pos):
        return pos[0] == len(self.board)-1

    def is__cell_SW(self, pos):
        return pos[1] == 0

    def make_move(self, pos):
        if self.board[pos[0]][pos[1]]>0:
            raise Exception("The move is not legal")
        else:
            self.board[pos[0]][pos[1]] = self.player_to_move.value
            self.empty_cells.remove((pos[0],pos[1]))
            # join black or red groups
            unions = None
            if self.player_to_move == Player.BLACK:
                unions = self.black_unions
            else:
                unions = self.red_unions
            sets_to_join = list()
            for neighbour in self.neighbours[pos]:
                if self.board[neighbour[0]][neighbour[1]] == self.player_to_move.value:
                    for s in unions: # find the union that has neigbour cell
                        if s not in sets_to_join and neighbour in s:
                            s.add(pos)
                            sets_to_join.append(s)
                            break
            if len(sets_to_join) == 0:
                s = set()
                s.add(pos)
                unions.append(s) #TODO: continue here
            if len(sets_to_join) > 1:
                union_of_sets = set().union(*sets_to_join)
                unions =  [el for el in unions if el not in sets_to_join]
                unions.append(union_of_sets)
            if self.player_to_move == Player.BLACK:
                self.black_unions = unions
            else:
                self.red_unions = unions
            if self.player_to_move == Player.BLACK:
                self.player_to_move = Player.RED
            else:
                self.player_to_move = Player.BLACK

    def get_legal_moves(self):
        return [pos for pos in self.empty_cells]

    def generate_successor_board_states(self):
        current_board_state = deepcopy(self.board)
        board_states = list()
        for pos in self.empty_cells:
            board = deepcopy(current_board_state)
            board[pos[0]][pos[1]] = self.player_to_move.value
            board_states.append(board)
        return board_states



def main() -> None:
    game = Game(3)
    game.make_move((1,0)) # black
    game.make_move((0,0)) # red
    game.make_move((2,0)) # black
    game.make_move((0,2)) # red
    game.make_move((1,2)) # b
    game.make_move((2,2)) # red
    game.make_move((1,1)) # b





if __name__ == '__main__':
    main()

