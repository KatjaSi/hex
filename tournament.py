from typing import List
from policy.target_policy import ANET
from hexgame import HexGame, HexStateManager

anet_1 = ANET.load("anets/anet0.h5", is_pipeline=False)
anet_2 = ANET.load("anets/anet50.h5", is_pipeline=False)
anet_3 = ANET.load("anets/anet100.h5", is_pipeline=False)
anet_4 = ANET.load("anets/anet150.h5", is_pipeline=False)
anet_5 = ANET.load("anets/anet196.h5", is_pipeline=False)



class Tournament:

    #def __init__(self, anets: List[ANET]) -> None:
     #   self.anets = anets
      #  self.anet_statistics = [Statistics(anet) for anet in anets]

    @staticmethod
    def play_one_game(player1: ANET, player2: ANET,  game: HexGame):
        # player1 is black player2 is red
        target_policy_1 = player1.target_policy
        target_policy_2 = player2.target_policy

        while not game.is_game_finished():
            if game.get_player_to_move() == 1:
                move = target_policy_1(state=game.state, actions=game.get_legal_moves())
            else:
                move = target_policy_2(state=game.state, actions=game.get_legal_moves())
            game.make_move(move)

        return game.get_winner()

 
    @classmethod
    def play_series(cls, player_1:ANET, player_2:ANET, G=25, board_size=4):

        statistics = dict()
        statistics['player_1'] = {'wins': 0, 'loses': 0}
        statistics['player_2'] = {'wins': 0, 'loses': 0}

        for i in range(G):
            game = HexGame(board_size, player=i%2+1) # who starts
            winner = cls.play_one_game(player_1, player_2, game)
            if winner == 1:
                statistics['player_1']['wins'] +=1
                statistics['player_2']['loses'] +=1
            else:
                statistics['player_2']['wins'] +=1
                statistics['player_1']['loses'] +=1
        
        return statistics

    @classmethod
    def play_tournament(cls, anets: List[ANET], G:int, board_size:int):

        statistics = dict()
        for i in range(len(anets)):
            statistics[f'player_{i+1}'] = {'wins': 0, 'loses': 0}

        for i in range(len(anets)):
            for j in range(i):
                series_statistics = cls.play_series(anets[i], anets[j], G, board_size)
                statistics[f'player_{i+1}']['wins'] +=series_statistics['player_1']['wins']
                statistics[f'player_{i+1}']['loses'] +=series_statistics['player_1']['loses']
                statistics[f'player_{j+1}']['wins'] +=series_statistics['player_2']['wins']
                statistics[f'player_{j+1}']['loses'] +=series_statistics['player_2']['loses']
    
        return statistics



#statistics = Tournament.play_series(anet_1, anet_5, G=1000, board_size=3)
statistics = Tournament.play_tournament([anet_1, anet_2, anet_4, anet_4, anet_5], G=50, board_size=3)
print(statistics)