# Node class
class BaseNode:
    Animal, Attribute = range(2)

    def get_type(self):
        raise NotImplementedError("All classes implementing BaseNode should implement get_type()")
