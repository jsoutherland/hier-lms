## Model Card

### Overview

This is a first pass at a 768D model.

It is a 768D 6-layer ModernBERT HLM.

The vocabulary was limited to the first 256 ASCII characters,
plus special characters, including one for unknown characters.


### Training
This was trained using a MLM task, as described in the paper.

I used a combination of `wikipedia-20220301.en.hf` and `openwebtext` datasets.

This is only a partially trained model on 500k steps. 

It is a checkpoint from further into the training
run that produced the `-255k` variant. This model was trained on almost one complete epoch for
345k steps and then another 155k steps restarting from the beginning of the dataloader. A complete epoch would have totalled just over 346k steps.

On the second pass/epoch the collator
was modified to randomly replace some characters of the MLM-chosen token with 50% probability, independently.

TODO: performance comparisons.

### Speed
This is a slower model than usual because:
* It uses two transformers
* The tokenizer uses spacy
* The collator and tokenizer haven't been optimized and are using python vs Rust or C.
