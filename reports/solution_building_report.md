## Solution Bulding Report

Text detoxification can be represented as a sequence-to-sequence style transfer task. I was looking not only for detoxification solutions, but for overall style transfer models for text.

Since this task is not trivial, classical ML approaches and even not that complex deep ones will not give any feasible result. So, I decided to focus on attention approaches as a baseline.

Obtained research log can be found at [Notion.](https://shy-gold-119.notion.site/Research-log-2afb5e725054430b90ac6ddfc8697f0f?pvs=4)

## Baseline: Attention based approaches.

### Hypothesis 1: Encoder-Decoder approach

One of the most straightforward ways of solving style transfer, as a sequence-to-sequence task, is to '“translate" a source sentence into the target style using a supervised encoder-decoder model.

I used this [tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#training-and-evaluating) as a base to build the model and transformed it for the style-transfer task.

After the training I obtained following results:\
where '>' for input, '=' for real label and '<' for output.

```
> aw damn !
= aw hell !
< aw hell ! the shoe ! <EOS>

> get the fuck out of here
= get out of here
< get out of here now <EOS>

> i can t wait to get rid of him
= honestly i can t wait to get rid of phillip
< honestly i can t wait to get rid of phillip <EOS>

> he s crazy but he s talented
= twisted but talented
< he s mad but he s bones <EOS>

> a couple of dumb shows never hurt anybody
= a couple of rescuers haven t hurt anyone yet
< a couple of rescuers haven t hurt anyone yet yet <EOS>

> apart from the blood spatters there are chums
= some of the blood is smears not spatter
< some of the blood is smears not spatter <EOS>

```
As you can see, it works :). However, sometimes there are some random outliers or words dublication, that is not good.

### Hypothesis 2: More suffisticated models

I tried to come up with something more advanced, but failed several times. For state-of-art research I found work done by [Skolkovo researchers](https://aclanthology.org/2021.emnlp-main.629.pdf). So I decided to try their two approches and maybe improve them.

1. tried [CondBert](https://github.com/s-nlp/detox/tree/main/emnlp2021/style_transfer/condBERT). But faced some version incostistency error with fliar library. For some time I tried to find what's the problem, but importing morre libraries with specific versions just lead to more errors during the training process. When I tried to reformulate it I faced complex project structure, that I don't really understand.

2. tried [ParaGeDi](https://github.com/s-nlp/detox/tree/main/emnlp2021/style_transfer/paraGeDi). For this module I faced some strange errors about missing functions at the main GediAdapter class. Moreover, I went through their code and thought that this model is build on top of their previous classifiers and paraphrasers. So, there is not much I could really do except changing some parameters and using some other pretrained models.

So I gave up with this idea, these people are cool and my small knowledge in NLP field doesn't allow me to be at this level. But it was cool to see state-of-art working solutions on detoxification topic. There is always something to strive for :)

### Hypothesis 3: Tuning Text-To-Text Transfer Transformer (T5)

I decided not to overcomplicate the things and try lab example that you provided.
For the final solution I took pretrained t5-small model from [Hugging Face](https://huggingface.co/t5-small) which is lightweight and easy to tune and train even on small dataset. 

I had some problems with training it on Kaggle, so I haven't tried much variations from the original example. However, the model achieved a quite good result considering limited resources and execution time.

```
> what the fuck
< what the hell is it?

> let's get the fuck out of here
< let's get out of here.

> goddamn what the hell are you doing?
< what are you doing?

> you are such a pussy
< you're so a saxy.

> kicks our asses and steals all the coke.
< he's stealing all the coke.

> kicks our asses and steals all the coke.
< he's stealing all the coke.

> oh shit. okay.
< okay.

> i don't dare take the life of your rusty sons for nothing.
< i don't want to take the life of your rusty sons for nothing

```

Quite good perfomance on some random inputs. Seems more about paraphrasing than just word removal, that's good. Also no random outliers.

## Results

Finally, I chose tuned T5 as my final model. I evaluated its perfomance on [Roberta Classifier](https://huggingface.co/s-nlp/roberta_toxicity_classifier) and got mean toxicity score of 6e-08, whereas original data has mean 1e-3 score. That's a huge improvement!

Also just for visualization purposes, the following wordcloud shows most often detoxified words

![](figures\cloud.png "Fig. 2")
