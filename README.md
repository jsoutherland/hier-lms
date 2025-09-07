# Hierarchical Language Models

Pytorch implementation models based on the paper:

[From Characters to Words: Hierarchical Pre-trained Language Model for Open-vocabulary Language Understanding](https://arxiv.org/abs/2305.14571)

## Project Purpose
I've been interested in the approach of this paper for a while and couldn't
find an implementation anywhere.  I'm creating one, with some modifications of my own.

## Project Status
This is a work in progress.  I'm still training and experimenting.

I have models available, but:
* They are under-tuned and under-trained
* They are smaller than the model from the paper
* They use a smaller character vocabulary, which requires tokenizer changes
* The tokenization and collation code is somewhat slow
* The approach to unknown (UNK) characters is functional, but incomplete
* The code is not fully transformers library-compliant yet

Despite that:
* I have models smaller than distilbert, that perform well on downstream tasks
* I believe that this project will benefit other projects of mine, which require it to be open sourced.

## Dependencies
The main dependencies are listed in [requirements.txt](requirements.txt)

Additionally, the `en_core_web_sm` spacy model is used by the tokenizer.

## Usage
I haven't made this code fully transformers-compliant, but it's close.
The basics:
```python
model = HlmForSequenceClassification(model_path, num_labels=2)

tokenizer = HlmTokenizer()

data_collator = HlmDataCollatorForSequenceClassification(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_imdb["train"],
    eval_dataset=tokenized_imdb["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```