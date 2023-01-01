import spacy
from dataclasses import dataclass

@dataclass
class EntityRecognition:
    text: str
    entity: list = None
    
    @staticmethod
    def corpus():
        return spacy.load("en_core_web_trf")

    def convert(self):
        nlp = self.corpus()
        return nlp(self.text)
    
    def construct(self):
        ner = self.convert()
        self.entity = [(word.label_, word.text) for word in ner.ents]
        
    def __post_init__(self):
        self.construct()

