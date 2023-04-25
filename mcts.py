from __future__ import annotations
import numpy as np
import time
from statemanager import StateManager
from nimgame import NimStateManager, NimGame
from hexgame import HexStateManager, HexGame
from typing import Tuple, Callable
from copy import deepcopy
from random import randint
from policy.tree_policy import  max_tree_policy, random_tree_policy, min_tree_policy
from policy.target_policy import random_target_policy



class Node:

    def __init__(self, state: Tuple[int], parent: Node | None = None)  -> None:
        self.state = state
        self.parent = parent
        self.children = dict() # key is action and value is a state
        self.score = 0
      #1  self.Q = dict() # Q values_ key is action, value is value of that action from the node state
        self.visits=0
        self.is_expanded = False

    def is_leaf(self):
        return not self.is_expanded  #len(self.children)==0



class MCTS:

    def __init__(self, SM: object.__class__, state: Tuple[int], tree_policy: Tuple[Callable], target_policy: Callable, player: int=1, M=500) -> None:
        self.root = Node(state=state)
        self.sm = SM
        self.tree_policy_player1 = tree_policy[0]
        self.tree_policy_player2 = tree_policy[1]
        self.target_policy = target_policy
        self.player = player
        self.M = M

        self.expand_node(self.root)

    def expand_node(self, node: Node, limit: int | None = None) -> None:
        #successor_states = self.sm.generate_child_states(state=node.state, limit=limit)
        actions = self.sm.get_all_legal_actions(node.state)
        for action in actions:
            # check if action already exists
            if action not in node.children:
                child_state = self.sm.generate_child_state(node.state, action)
                child_node = Node(state=child_state, parent=node)
                node.children[action] = child_node
        node.is_expanded = True
    
    def rollout(self, leaf_node):
        """
        Leaf Evaluation: Estimating the value of a leaf node in the tree by doing a rollout simulation using the default
        policy from the leaf node’s state to a final state.
        """
        node = leaf_node
        state = node.state 
        node.visits += 1
        while not self.sm.is_game_finished(state):
            actions = self.sm.get_all_legal_actions(state)
            action = self.target_policy(state,actions)
            state = self.sm.generate_child_state(node.state, action)
            child_node = Node(state=state, parent=node)
           # node.children[action] = child_node not keeping the rollout nodes
            node=child_node
           # node.visits += 1
       # node.visits -= 1
        final_state = state
        if self.sm.get_winner(final_state) == self.player and self.player == 1:
            upd_score = 1
        else:
            upd_score = -1
        self.backpropagate(node, upd_score)

    def backpropagate(self, final_node: Node, score: int) -> None:
        final_node.score += score
        node = final_node
        while node.parent is not None:
            node = node.parent
            node.score += score
                
            
    def simulate(self):
        start_time = time.time()
        elapsed_time = 0
        if not self.root.is_expanded:
            self.expand_node(self.root) # expanding the root first, because i will need to choose one of 
        # the all possible actions from the root, so here it is more important to get all of them
        #for i in range(self.M):
         #   if i % 10 == 0:
          #      print(i)
        M = 0  
        while elapsed_time < 2:  # loop until 1 second has elapsed
            node = self.tree_search()
            if self.sm.is_game_finished(node.state):
                 self.rollout(node)
            else:     
                self.expand_node(node)
                actions = list(node.children.keys())
                #action = actions[randint(0, len(actions)-1)]  
                #      Tree search may go directly from tree-policy moves to a
                #      rollout, without expanding the leaf node
                state = node.state
                action = self.target_policy(state, actions)
                c = node.children[action]
                self.rollout(c)
            elapsed_time = time.time() - start_time  # calculate elapsed time
            M += 1
        print(M)

    def tree_search(self):
        node = self.root
        while not node.is_leaf():
            node.visits += 1 
            player = self.sm.get_player(node.state)
            if player == 1:
                tree_policy = self.tree_policy_player1
            else:
                tree_policy = self.tree_policy_player2
            node = tree_policy(node.children)
        return node
    
    def get_move(self):
        ### visits reflect the Q value
        action =  max(self.root.children.keys(), key=lambda k: self.root.children[k].visits+1)
        return action

    def get_visits_distributions(self):
        """
        Visists distributions are returned as two lists: moves and visits.
        Moves are the list of all moves, e.g. (0,0), (0,1), (0,2) ....., (6,6)
        Visists is the list of visits, e.g. 12,2323, .... 23. 
        The number of visits corresponding to the move at index i in the list of moves is visits[i]
        Only the move corresponding to legal actions will be return
        """
        D = dict()
        for child in self.root.children:
            D[child] = self.root.children[child].visits 
        return D

    def reset_root(self, action):
        self.root = self.root.children[action]
        #prune
        self.root.parent = None

def main() -> None:
    game = HexGame(3)
    #s_init = NimStateManager.generate_initial_state(10,3)
    s_init = HexStateManager.generate_initial_state(size=3)
    mcts = MCTS(SM=HexStateManager, state=s_init, tree_policy=(max_tree_policy, min_tree_policy), target_policy=random_target_policy)
    while not game.is_game_finished():
        print("Player 1s turn")
        mcts.simulate()
        print("simulating..")
        #for child in mcts.root.children.keys():
         #   print(f"{child} has score/visits {mcts.root.children[child].score}/{mcts.root.children[child].visits}")
        print(mcts.get_visits_distributions()) 
        move = mcts.get_move()
        game.make_move(move)
        print(game.state)
        if game.is_game_finished():
            break
        mcts.reset_root(move)
        # for hex only
        user2_move = input("Your turn. Enter yout move: ")
        user2_move = user2_move.split(',')
        user2_move = (int(user2_move[0]), int(user2_move[1]))
        action = user2_move
        #### end for hex only
        #action = int(user2_move)
        game.make_move(action)
        mcts.expand_node(mcts.root)
        mcts.reset_root(action)
        print(game.state)
    print(f"Game over. the winner is player {game.get_winner()}")

     


if __name__ == '__main__':
    main()
    
    