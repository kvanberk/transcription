import whisper
from dataclasses import dataclass

@dataclass
class Whisper:
    model: str
    language: str

    def initiate(self):
        return whisper.load_model(self.model)

    def transcript(self, file):
        return self.initiate().transcribe(file)