from decision_layer.encoders import BaseEncoder
from decision_layer.schema import Decision


class DecisionLayer:
    def __init__(self, encoder: BaseEncoder, decisions: list[Decision]):
        self.encoder = encoder
        self.decisions = decisions
        if decisions:
            pass


    def add(self, decision: Decision):
        pass
        embeds = encoder(te)