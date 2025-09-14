## Model Card

### Overview

This is a first pass at a 768D model.

It is a 768D 6-layer ModernBERT HLM.

The vocabulary was limited to the first 256 ASCII characters,
plus special characters, including one for unknown characters.


### Training
This was trained using a MLM task, as described in the paper.

I used a combination of `wikipedia-20220301.en.hf` and `openwebtext` datasets.

This is only a partially trained model on 255k steps (less than one epoch). 

That said, this model beats the 768D [distilbert-base-cased](distilbert/distilbert-base-cased)
model on downstream tasks involving heavy text corruption, such as `imdb` with text corruption as described
in the paper.

When fine-tuning with this base model, it reaches F1 of 0.88 while the best I have achieved with distilbert-base-cased is 0.78.

### Speed
This is a slower model than usual because:
* It uses two transformers
* The tokenizer uses spacy
* The collator and tokenizer haven't been optimized and are using python vs Rust or C.
