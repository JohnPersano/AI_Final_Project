# Node class
class Node:

    def __init__(self, name="", parent=""):
        self.name = name
        self.children = []
        self.parent = parent

    def append(self, child="", parent=""):
        if child != "":
            self.children.append(child)
        if parent != "":
            self.parent = parent

    def remove_child(self, child):
        if child.name in self.children:
            self.children.remove(child.name)

    def print(self):
        print("<Node name={} children={} parent=[{}]>".format(self.name, self.children, self.parent))
