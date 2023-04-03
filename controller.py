import time
from hexgame import HexGame
from gui import Hex
from mcts import MCTS
from policy.target_policy import ANET

class Controller:

    def __init__(self, game_view:Hex, game_model:HexGame, mcts: MCTS):
        self.game_view = game_view 
        self.game_model = game_model
        game_view.add_listener(self) # listen to guis click event
        self.mcts = mcts

    def start_game(self):
        self.game_view.game_loop()

    def on_generate_move(self):
        if self.game_model.get_player_to_move() == 1:
            self.mcts.simulate()
            move = self.mcts.get_move()
            item = self.board_index_to_item(move)
            self.game_view.canvas.itemconfig(item, fill='black')
            
            self.game_model.make_move(move)
            self.mcts.reset_root(move)
            self.mcts.M = int(self.mcts.M*1.2)
            self.game_view.player_label.config(text = "RED's turn")
            if self.game_model.black_is_won():
                    self.game_view.won_label.config(text = "BLACK won!")
                    self.game_view.end_game()
                   # return

            #self.on_generate_move()
        else:
            self.mcts.simulate()
            move = self.mcts.get_move()
            item = self.board_index_to_item(move)
            self.game_view.canvas.itemconfig(item, fill='red')
            self.game_model.make_move(move)
            self.mcts.reset_root(move)
            self.mcts.M = int(self.mcts.M*1.2)
            self.game_view.player_label.config(text = "BLACK's turn")
            if self.game_model.red_is_won():
                    self.game_view.won_label.config(text = "RED won!")
                    self.game_view.end_game()
                    #return

            #self.on_generate_move()

                
    def on_canvas_click(self, event, item):
        board_index = self.item_to_board_index(item[0])
        #if self.game_model.get_player_to_move() == 1: #Player.BLACK:
         #   self.game_model.make_move(board_index)
          #  self.mcts.reset_root(board_index) # to be removed maybe
           # self.game_view.canvas.itemconfig(item, fill='black')
            #if self.game_model.black_is_won():
             #   self.game_view.won_label.config(text = "BLACK won!")
              #  self.game_view.end_game()
            #self.game_view.player_label.config(text = "RED's turn")
        #else:
        if self.game_model.get_player_to_move() == 2:
            self.game_model.make_move(board_index)
            self.mcts.reset_root(board_index) # to be removed maybe
            self.game_view.canvas.itemconfig(item, fill='red')
            if self.game_model.red_is_won():
                self.game_view.won_label.config(text = "RED won!")
                self.game_view.end_game()
                return
            self.game_view.player_label.config(text = "BLACK's turn")

            # generate pc move
            self.on_generate_move()
        

    def item_to_board_index(self, item_index):
        i = (item_index-1)//self.game_model.size
        j = (item_index-1)%self.game_model.size
        return (i,j)
    
    def board_index_to_item(self, board_index):
        i, j = board_index
        item = i*self.game_model.size + j + 1
        return item
    
    def make_move(self, move):
        self.game_model.make_move(move)



# Anet controller
class ANET_Controller:

    def __init__(self, game_view:Hex, game_model:HexGame, anet:ANET):
        self.game_view = game_view 
        self.game_model = game_model
        self.anet = anet
        self.game_view.add_listener(self) 

    def start_game(self):
        self.game_view.game_loop()

    def on_generate_move(self):
        if self.game_model.get_player_to_move() == 1:
            move = self.anet.target_policy(state=self.game_model.state, actions=self.game_model.get_legal_moves())
            item = self.board_index_to_item(move)
            self.game_view.canvas.itemconfig(item, fill='black')
            self.game_model.make_move(move)
            self.game_view.player_label.config(text = "RED's's turn")
            if self.game_model.black_is_won():
                    self.game_view.won_label.config(text = "BLACK won!")
                    self.game_view.end_game()
                    return

        else:
            move = self.anet.target_policy(state=self.game_model.state, actions=self.game_model.get_legal_moves())
            item = self.board_index_to_item(move)
            self.game_view.canvas.itemconfig(item, fill='red')
            self.game_model.make_move(move)
            self.game_view.player_label.config(text = "BLACK's turn")
            if self.game_model.red_is_won():
                    self.game_view.won_label.config(text = "RED won!")
                    self.game_view.end_game()
                    return

    def on_canvas_click(self, event, item):
        board_index = self.item_to_board_index(item[0])
        #if self.game_model.get_player_to_move() == 1: #Player.BLACK:
         #   self.game_model.make_move(board_index)
          #  self.mcts.reset_root(board_index) # to be removed maybe
           # self.game_view.canvas.itemconfig(item, fill='black')
            #if self.game_model.black_is_won():
             #   self.game_view.won_label.config(text = "BLACK won!")
              #  self.game_view.end_game()
            #self.game_view.player_label.config(text = "RED's turn")
        #else:
        if self.game_model.get_player_to_move() == 2:
            self.game_model.make_move(board_index)
            self.game_view.canvas.itemconfig(item, fill='red')
            if self.game_model.red_is_won():
                self.game_view.won_label.config(text = "RED won!")
                self.game_view.end_game()
                return
            self.game_view.player_label.config(text = "BLACK's turn")

            # generate pc move
            self.on_generate_move()
        

    def item_to_board_index(self, item_index):
        i = (item_index-1)//self.game_model.size
        j = (item_index-1)%self.game_model.size
        return (i,j)
    
    def board_index_to_item(self, board_index):
        i, j = board_index
        item = i*self.game_model.size + j + 1
        return item
    
    def make_move(self, move):
        self.game_model.make_move(move)