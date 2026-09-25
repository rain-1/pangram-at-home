"""Offset-preserving windows and supervision for the binary span pilot."""
import json
from pathlib import Path


def window_starts(n: int, size: int = 512, stride: int = 256) -> list[int]:
    if not 0 < stride <= size:
        raise ValueError("Require 0 < stride <= window size")
    if n <= size:
        return [0]
    return sorted(set([*range(0, n-size+1, stride), n-size]))


def encode_document(row, tokenizer):
    encoded = tokenizer(row["text"], add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")
    labels = []
    for start, end in offsets:
        matches = {span["label"] for span in row["spans"]
                   if span["start"] < end and span["end"] > start}
        labels.append(next(iter(matches)) if len(matches) == 1 else -100)
    return encoded["input_ids"], offsets, labels


def token_window(ids, labels, repeat2=True):
    if len(ids) != len(labels):
        raise ValueError("Token and label lengths differ")
    return {"input_ids": ids + ids if repeat2 else ids,
            "attention_mask": [1] * (len(ids) * (2 if repeat2 else 1)),
            "labels": [-100] * len(ids) + labels if repeat2 else labels}


class SpanDataset:
    def __init__(self, path: Path, tokenizer, size=512, stride=256, repeat2=True):
        self.rows = [json.loads(line) for line in path.read_text().splitlines()]
        self.windows = []
        for row in self.rows:
            ids, _, labels = encode_document(row, tokenizer)
            for start in window_starts(len(ids), size, stride):
                self.windows.append(token_window(ids[start:start+size], labels[start:start+size], repeat2))

    def __len__(self): return len(self.windows)

    def __getitem__(self, index): return self.windows[index]
