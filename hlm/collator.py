# Copyright 2025 Joshua Southerland
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .tokenizer import HlmTokenizer


class HlmDataCollatorForSequenceClassification:
    """ Example of how to collate data for downstream tasks. """
    def __init__(self, tokenizer: "HlmTokenizer", max_seq_len: int = 512):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __call__(self, batch):
        # TODO: speed this up
        batch_size = len(batch)
        max_words = 0
        max_chars = 0
        for sample in batch:
            input_ids = sample['input_ids']
            word_count = len(input_ids)
            if word_count > max_words:
                max_words = word_count
            for word in input_ids:
                if len(word) > max_chars:
                    max_chars = len(word)

        max_words = min(self.max_seq_len, max_words) # TODO: do truncation elsewhere

        input_ids = torch.zeros((batch_size, max_words, max_chars), dtype=torch.long)
        for ind, sample in enumerate(batch):
            sample_input_ids = sample['input_ids']
            for word_ind, word in enumerate(sample_input_ids):
                if (word_ind + 1) >= self.max_seq_len:
                    break # TODO: truncation elsewhere
                num_chars = len(word)
                input_ids[ind, word_ind, :num_chars] = torch.LongTensor(word)

        labels = torch.LongTensor([s["label"] for s in batch])
        attention_mask = (input_ids != self.tokenizer.pad_id).long()

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
        return out
