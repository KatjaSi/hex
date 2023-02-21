from hexgame import HexGame
from gui import Hex
from controller import Controller

game_model = HexGame(5)
game_view = Hex(size=5)
controller = Controller(game_view=game_view, game_model=game_model)

controller.start_game()