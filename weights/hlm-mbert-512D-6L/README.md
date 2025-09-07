## Model Card

### Overview

This is a first pass at adding a smaller model. 

It is a 512D 6-layer ModernBERT HLM.

The vocabulary was limited to the first 256 ASCII charaters,
plus special characters, including one for unknown characters.


### Training
This was trained using a MLM task, as described in the paper.

I used a combination of `wikipedia-20220301.en.hf` and `openwebtexts` datasets.

This is only a partially trained model on 146k steps (less than one epoch). 
I will likely update these weights after I evaluate model that has been trained further.

That said, this model  beats the larger distilbert-base-cased model on downstream tasks 
involving heavy text corruption, such as `imdb` with text corruption as described
in the paper.

This model reaches F1 of 0.86 while the best I have achieved with distilbert-base-cased is 0.78.