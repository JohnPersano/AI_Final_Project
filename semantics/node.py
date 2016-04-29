

class Node:

    def __init__(self, type=""):
        self.type = type

    def get_type(self):
        raise NotImplementedError("All nodes should implement get_type()")
