from game import Game, Player
from gui import Hex

class Controller:

    def __init__(self, game_view, game_model):
        self.game_view = game_view 
        self.game_model = game_model
        game_view.add_listener(self) # listen to guis click event

    def start_game(self):
        self.game_view.game_loop()

    def on_canvas_click(self, event, item):
        board_index = self.item_to_board_index(item[0])
        if self.game_model.player_to_move == Player.BLACK:
            self.game_model.make_move(board_index)
            self.game_view.canvas.itemconfig(item, fill='black')
            if self.game_model.black_is_won():
                self.game_view.won_label.config(text = "BLACK won!")
                self.game_view.end_game()
            self.game_view.player_label.config(text = "RED's turn")
        else:
            self.game_model.make_move(board_index)
            self.game_view.canvas.itemconfig(item, fill='red')
            if self.game_model.red_is_won():
                self.game_view.won_label.config(text = "RED won!")
                self.game_view.end_game()
            self.game_view.player_label.config(text = "BLACK's turn")
        

    def item_to_board_index(self, item_index):
        i = (item_index-1)//self.game_model.size
        j = (item_index-1)%self.game_model.size
        return (i,j)