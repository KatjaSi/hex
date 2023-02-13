from game import Game
from gui import Hex

class Controller:

    def __init__(self, game_view, game_model):
        self.game_view = game_view 
        self.game_model = game_model
        game_view.add_listener(self) # listen to guis click event

    def start_game(self):
        self.game_view.game_loop()

    def on_canvas_click(self, event, item):
        print(item)