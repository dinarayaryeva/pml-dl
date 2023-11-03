# Final Solution Report

## Introduction

Text detoxification of text is a relatively new style transfer task.
The style transfer task is generally understood as rewriting of text with the same content and with altering of one or several attributes which constitute the "style". Despite the goal of preserving the content, in many cases changing the style attributes changes the meaning of a sentence significantly.

1. So in fact the goal of many style transfer models is to transform a sentence into a somewhat similar sentence of a different style on the same topic.

2. Detoxification needs better preservation of the original meaning than many other style transfer tasks, such as sentiment transfer, so it should be performed differently.

### Style-transfer

## Data analysis

### ParaNMT

The dataset is a subset of the ParaNMT corpus (50M sentence pairs). The filtered ParaNMT-detox corpus (500K sentence pairs) can be downloaded from [here](https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).


| Column | Type | Discription | 
| ----- | ------- | ---------- |
| reference | str | First item from the pair | 
| ref_tox | float | toxicity level of reference text | 
| translation | str | Second item from the pair - paraphrazed version of the reference|
| trn_tox | float | toxicity level of translation text |
| similarity | float | cosine similarity of the texts |
| lenght_diff | float | relative length difference between texts |

During the data exploration, I got:

1. No missed values inside the dataset.
2. Samples have a huge similarity, from 0.6 to 0.95.
3. Length difference between reference and translation is low.
4. Value for toxicity score is normalized to from 0 to 1 for each sample.
5. The dataset is imbalanced in terms of toxicity level. Most of the samples either too neutral or too toxic. "trn_tox" is mostly neutral, while "ref_tox" tends to have higher toxicity score.
6. All the samples with difference between "trn_tox" and "ref_tox" below 0.5 were not included into the dataset.

![](figures\heatmap.png "Fig. 2")

We see that ref_tox and trn_tox are in huge dependency with each other. All other fields cause no effect on toxicity fields, therefore they will not be used furher considerations.

### Preprocessing

I decided to make it as minimal as I can, because it would be better to use just autotokenizers for deep learning models and apply them further, then to try to learn language structure and toxicity features from too clean data.

The one thing I did is to remove commas, otherwise transformation from tsv to csv lead to many errors. I tried to fix it without removing commas and I failed.

## Model Specification

### T5

The proposed approach is to finetune T5(Text-To-Text Transfer Transformer) model on style-transfer task. For this purpose we will be using HuggingFace transformers. For the example purpose we select as model checkpoint the smallest transformer in T5 family - t5_small.

## Training Process

In order to train t5-small I used Seq2Seq pretrained tokenizers, data collators and trainers from transformers library.

### Hyperparameters
```python
batch_size = 32
learning_rate=2e-5
weight_decay=0.01
save_total_limit=3
num_train_epochs=10
```
### Metrics

Style transfer models need to change the style,
preserve content and produce a fluent text. These
parameters are often inversely correlated, so we
need a compound metric to find a balance between
them. So it's necessary to choose appropriate metrics.

The main metric is for training was BLEU score. The BLEU score measures the similarity of the machine-translated text to a set of high quality reference translations. The BLEU metric is calculates using n-grams.

Unfortunately, I had no GPU time on Kaggle and also some random problems, so I had to train the model on Colab, that vanishes runtime. That's why for now I have only last checkpoint, not the final model save, I just couldn't find it on my drive. You can download it from [here](https://drive.google.com/drive/folders/1gHOxODVGO1xua27KlHhRuAf6_kR_zl-R?usp=sharing).

## Results

I evaluated its perfomance on [Roberta Classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier) and got mean toxicity score of 6e-08, whereas original data has mean 1e-3 score. That's a huge improvement!

Also just for visualization purposes, the following wordcloud shows most often detoxified words

![](figures\cloud.png "Fig. 2")