from pathlib import Path
from tokenizers import Tokenizer as Backend, models, trainers, pre_tokenizers, decoders


class Tokenizer:
    def __init__(self, backend):
        self.backend = backend
        self.eos_id = backend.token_to_id('<|endoftext|>')
        if self.eos_id is None:
            raise ValueError('Tokenizer missing document boundary token')

    @classmethod
    def train(cls, texts, vocab_size=16384):
        if vocab_size < 257:
            raise ValueError('Byte-level tokenizer needs at least 257 entries')
        backend = Backend(models.BPE())
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        backend.train_from_iterator(texts, trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=['<|endoftext|>'], initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
        return cls(backend)

    @property
    def vocab_size(self):
        return self.backend.get_vocab_size()

    def encode(self, text):
        return self.backend.encode(text, add_special_tokens=False).ids

    def decode(self, ids):
        return self.backend.decode(list(ids), skip_special_tokens=False)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.backend.save(str(path))

    @classmethod
    def load(cls, path):
        return cls(Backend.from_file(str(path)))
