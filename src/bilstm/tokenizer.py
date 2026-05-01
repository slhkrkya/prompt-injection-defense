import json
import re
from pathlib import Path

PAD_IDX = 0
UNK_IDX = 1


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


class WordTokenizer:
    def __init__(self, vocab: dict[str, int] | None = None):
        self.vocab = vocab or {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}

    @classmethod
    def build_vocab(cls, texts: list[str], max_vocab: int = 10_000) -> "WordTokenizer":
        from collections import Counter

        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))
        vocab: dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
        for token, _ in counter.most_common(max_vocab - 2):
            vocab[token] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int = 512) -> list[int]:
        tokens = _tokenize(text)[:max_len]
        return [self.vocab.get(t, UNK_IDX) for t in tokens] or [PAD_IDX]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.vocab, ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "WordTokenizer":
        vocab = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: int(v) for k, v in vocab.items()})

    def __len__(self) -> int:
        return len(self.vocab)
