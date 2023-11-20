### initialize the strategy class
class Strategy:
    def __init__(self, name, strategy, params):
        self.name = name
        self.strategy = strategy
        self.params = params

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name