from policy.target_policy import ANET

from typing import List
from policy.target_policy import ANET
from hexgame import HexGame, HexStateManager
from rbuf import RBUF

anet_0 = ANET(board_size=5, method="use-distribution")
anet_0.method = 'most-probable'
anet_0.save("anets_lm/anet_00.h5")
anet_0 = ANET.load("anets_lm/anet_00.h5", is_pipeline=False)
anet_1 = ANET.load("anets_lm/anet_00.h5", is_pipeline=False)

#anet_2 = ANET.load("anets_4x4_seq/anet51.h5", is_pipeline=False)
anet_2 = ANET.load("anets_lm/anet50.h5")
#anet_3 = ANET.load("anets_4x4_seq/anet99.h5", is_pipeline=False)
anet_3 = ANET.load("anets_lm/anet100.h5")
#anet_3.eps = 0.0
#anet_4 = ANET.load("anets_4x4_seq/anet150.h5", is_pipeline=False)
anet_4 = ANET.load("anets_lm/anet150.h5")
#anet_4.eps = 0.0
#anet_5 = ANET.load("anets_4x4_seq/anet201.h5", is_pipeline=False)
#anet_5.eps = 0.0
anet_6 = ANET.load("anets_lm/anet200.h5")
#anet_6.eps = 0.0
#anet_4 = ANET.load("anets_lm/anet150.h5", is_pipeline=False)
#anet_5 = ANET.load("anets/anet196.h5", is_pipeline=False)



class Tournament:

    #def __init__(self, anets: List[ANET]) -> None:
     #   self.anets = anets
      #  self.anet_statistics = [Statistics(anet) for anet in anets]

    @staticmethod
    def play_one_game(player1: ANET, player2: ANET,  game: HexGame):
        # player1 is black player2 is red
        target_policy_1 = player1.target_policy
        target_policy_2 = player2.target_policy

        # first move is random
        player1.eps = 0.99
        player2.eps = 0.99
        if game.get_player_to_move() == 1:
            move = target_policy_1(state=game.state, actions=game.get_legal_moves())
        else:
            move = target_policy_2(state=game.state, actions=game.get_legal_moves())
        game.make_move(move)
        player1.eps = 0.0
        player2.eps = 0.0
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
            player = i%2+1 #who starts?
            game = HexGame(board_size, player=player) # who starts
            winner = cls.play_one_game(player_1, player_2, game)
            if winner == 1: #black
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

#anet_1.eps = 0.9
anet_2.eps = 0
#for _ in range(10):
 #   anet_0 = ANET(board_size=5)
  #  anet_0.method = "use-distribution"
   # statistics = Tournament.play_series(anet_4, anet_0, G=100, board_size=5)
    #print(statistics)
statistics = Tournament.play_series(anet_6, anet_0, G=100, board_size=5)
print(statistics)    
statistics = Tournament.play_series(anet_6, anet_2, G=100, board_size=5)
print(statistics)
statistics = Tournament.play_series(anet_6, anet_3, G=100, board_size=5)
print(statistics)
statistics = Tournament.play_series(anet_6, anet_4, G=100, board_size=5)
print(statistics)
statistics = Tournament.play_series(anet_6, anet_6, G=100, board_size=5)
print(statistics)

#rbuf = RBUF.load('rbuf4x4', board_size=4)

