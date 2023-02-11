"""
Module for game board and gam cell representation
"""
from enum import Enum

class Player(Enum):
    BLACK = 1,
    RED = 2
 

class Game:

    class Cell:

        def __init__(self, pos_on_board, board_size):
            self.neighbours = list()
            self.state = (0,0)
            self.board_size=board_size
            neighbour_indices = [(-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0)]
            self.pos_on_board = pos_on_board
            for ind in neighbour_indices:
                if (self.pos_on_board[0]+ind[0] >= 0 and self.pos_on_board[0]+ind[0] < board_size \
                    and self.pos_on_board[1]+ind[1] >= 0 and self.pos_on_board[1]+ind[1] < board_size):
                    self.neighbours.append((self.pos_on_board[0]+ind[0], self.pos_on_board[1]+ind[1]))
            self.board_size = board_size

        
        def is_NW(self):
            return self.pos_on_board[0] == 0

        def is_NE(self):
            return self.pos_on_board[1] == self.board_size-1

        def is_SE(self):
            return self.pos_on_board[0] == self.board_size-1

        def is_SW(self):
            return self.pos_on_board[1] == 0

        def is_empty(self):
            return self.state == (0,0)
        
        def fill_cell(self, player):
            self.state = (1,0) if player == Player.BLACK else (0,1)

        def occupied_by_player(self, player):
            if player == Player.BLACK:
                return self.state == (1, 0)
            return self.state == (0, 1)
        
        def __str__(self):
            return str(self.pos_on_board)


    def __init__(self, size, board=None):
        self.size = size
        # board of cells where each cell knows its position on the board and position of its neighbours on the board
        if board is None:
            self.board = [[self.Cell((i,j), size) for j in range(size)] for i in range(size)]
            self.empty_cells= [self.board[i][j] for i in range(size) for j in range(size)]
            self.player_to_move = Player.BLACK
        else:
            self.board = board
            self.empty_cells= [self.board[i][j] for i in range(size) for j in range(size) if self.board[i][j].is_empty()]
            player_to_move = (self.size**2 - len(self.empty_cells)) % 2 + 1
            if player_to_move == 1:
                self.player_to_move = Player.BLACK
            else:
                self.player_to_move = Player.RED
        self.players = [Player.BLACK, Player.RED] # black wants to connect NW && SE, red - SW && NE
        self.black_unions = list()
        self.red_unions = list()

    def show_board(self):
        board_str = ""
        for i in range(self.size):
            for j in range(self.size):
                board_str += str(self.board[i][j])
            board_str += "\n"
        return board_str

    def get_player_to_move(self):
        return self.players[self.player_to_move-1]

    def make_move(self, pos):
        cell = self.board[pos[0]][pos[1]]
        if not cell.is_empty:
            raise Exception("The move is not legal")
        else:
            cell.fill_cell(self.player_to_move)
            self.empty_cells.remove(cell)
            # join black or red groups
            unions = None
            if self.player_to_move == Player.BLACK:
                unions = self.black_unions
            else:
                unions = self.red_unions
            sets_to_join = list()
            for neighbour in cell.neighbours:
                neighbour_cell = self.board[neighbour[0]][neighbour[1]]
                if neighbour_cell.occupied_by_player(self.player_to_move):
                    for s in unions: # find the union that has neigbour cell
                        if s in sets_to_join:
                            break 
                        if neighbour_cell in s:
                            s.add(cell)
                            sets_to_join.append(s)
                            break
            if len(sets_to_join) == 0:
                s = set()
                s.add(cell)
                unions.append(s) #TODO: continue here
            if len(sets_to_join) > 1:
                union_of_sets = set().union(*sets_to_join)
                unions =  [el for el in unions if el not in sets_to_join]
                unions.append(union_of_sets)
            if self.player_to_move == Player.BLACK:
                self.player_to_move = Player.RED
            else:
                self.player_to_move = Player.BLACK

    def __join_unions__(self, unions): # unions is a list of unions of cells to be joined
        pass

    def get_legal_moves(self):
        return [cell.pos_on_board for cell in self.empty_cells]

    def generate_successor_board_states(self):
        board_states = list()
        for cell in self.empty_cells:
            board = [[self.Cell((i,j), len(self.board)) for j in range(self.size)] for i in range(self.size)]
            for i in range(self.size):
                for j in range(self.size):
                    board[i][j].state = self.board[i][j].state
            board[cell.pos_on_board[0]][cell.pos_on_board[1]].fill_cell(self.player_to_move)
            board_states.append(board)
        return board_states



def main() -> None:
    game = Game(3)
    # TODO: check the logic of unions
    game.make_move((1,0)) # black
    game.make_move((0,0)) # red
    game.make_move((2,0)) # black
    game.make_move((0,2)) # red
    game.make_move((1,2)) # b
    game.make_move((2,2)) # red
    game.make_move((1,1)) # b
    print(game.black_unions)


if __name__ == '__main__':
    main()

