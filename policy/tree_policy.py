from __future__ import annotations
from random import randint
from typing import List
import math
#from mcts import Node

def random_tree_policy(children):
    """
    children are to be the children of the node
    """
    return children[list(children.keys())[randint(0, len(children)-1)]]

def max_tree_policy(children):
    """
    the policy for the player who wants to maximize the score
    """
    return max(children.values(), key=lambda child: Q(child) + UCT(child)) 

def min_tree_policy(children):
    """
    the policy for the player who wants to maximize the score
    """
    return min(children.values(), key=lambda child: Q(child)-UCT(child)) 

def UCT(child_node): # prior probability that should come from the actor NN ?
    N_s_a = child_node.visits
    N_s = child_node.parent.visits
    c=2
    return c*math.sqrt(math.log(N_s)/(1+N_s_a))

def Q(child_node):
    return child_node.score /(child_node.visits+1) 