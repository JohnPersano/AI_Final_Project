"""
CSCI 6660 Final Project

Author: John Persano
Date:   04/27/2016
"""


class Node:
    """
    Base node class for the ObjectNode and RelationNode to implement.
    """
    def __init__(self, type=""):
        self.type = type

    def get_type(self):
        raise NotImplementedError("All nodes should implement get_type()")
