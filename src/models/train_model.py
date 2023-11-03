# necessary imports
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer
from transformers import AutoTokenizer
import warnings
from datasets import load_metric
import transformers
import datasets
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

"""## Dataset loading"""


dir_path = sys.path[1] + '/data/interim/'

train_df = pd.read_csv(dir_path + 'train.csv', index_col=0)
train_df.reset_index(drop=True, inplace=True)

val_df = pd.read_csv(dir_path + 'validate.csv', index_col=0)
val_df.reset_index(drop=True, inplace=True)

"""## T5 tuning"""

# selecting model checkpoint
model_checkpoint = "t5-small"

# setting random seed for transformers library
transformers.set_seed(42)

# Load the BLUE metric
metric = load_metric("sacrebleu")


# we will use autotokenizer for this purpose
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# prefix for model input
prefix = "make sentence neutral:"

max_input_length = 128
max_target_length = 128

target = "target"
source = "source"


def preprocess_function(example):

    inputs = [prefix + ex for ex in example[source]]
    targets = [ex for ex in example[target]]
    model_inputs = tokenizer(
        inputs, max_length=max_input_length, padding=True, truncation=True)
    # Setup the tokenizer for targets
    labels = tokenizer(targets, max_length=max_target_length,
                       padding=True, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = datasets.Dataset.from_pandas(train_df, split="train")
val_dataset = datasets.Dataset.from_pandas(val_df, split="train")

train_dataset_map = train_dataset.map(preprocess_function, batched=True)
val_dataset_map = val_dataset.map(preprocess_function, batched=True)


# create a model for the pretrained model
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# defining the parameters for training
batch_size = 32
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-{source}-to-{target}",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    report_to='tensorboard',
)

# instead of writing collate_fn function we will use DataCollatorForSeq2Seq
# similarly it implements the batch creation for training

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# simple postprocessing for text

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels

# compute metrics function to pass to trainer


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


# instead of writing train loop we will use Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=train_dataset_map,
    eval_dataset=val_dataset_map,
    data_collator=data_collator,
    tokenizer=tokenizer,


    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model(sys.path[1] + "/models/best")
