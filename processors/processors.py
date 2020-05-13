import torch 

from mmf.datasets.processors import FastTextProcessor
from mmf.common.registry import registry


@registry.register_processor("fasttext_sentence_vector")
class FastTextSentenceVectorProcessor(FastTextProcessor):
    def __call__(self, item):
        self._load_fasttext_model(self.model_file)
        if "text" in item:
            text = item["text"]
        elif "tokens" in item:
            text = " ".join(item["tokens"])

        sentence_vector = torch.tensor(
            self.model.get_sentence_vector(text),
            dtype=torch.float
        )
        return {
            "text": sentence_vector
        }
    
    # Make dataset builder happy, return a random number
    def get_vocab_size(self):
        return 100