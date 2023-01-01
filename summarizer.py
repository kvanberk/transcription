from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from dataclasses import dataclass

@dataclass
class LexSummarizer:
    text: str
    sentence: int
    language: str = "english"
    summary: str = ""

    @staticmethod
    def initiate():
        return LexRankSummarizer()

    @staticmethod
    def parser(text, language):
        return PlaintextParser.from_string(text,Tokenizer(language))

    def summarizer(self):
        lex_summarizer = self.initiate()
        parser = self.parser(self.text, self.language)
        return lex_summarizer(parser.document, self.sentence)

    def construct(self, summary):
        for sentence in summary:
            self.summary += f"{str(sentence)} "

    def __post_init__(self):
        summary = self.summarizer()
        self.construct(summary)


