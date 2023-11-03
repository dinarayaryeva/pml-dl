from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import pandas as pd
import sys

dir_path = sys.path[1] + 'data/interim/'

test_df = pd.read_csv(dir_path + 'test.csv', index_col=0)
test_df.reset_index(drop=True, inplace=True)


model_checkpoint = "t5-small"

# we will use autotokenizer for this purpose
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# prefix for model input
prefix = "make sentence neutral:"

# loading the model and run inference for it

model = AutoModelForSeq2SeqLM.from_pretrained(
    sys.path[1] + 'models/checkpoint-10000')
# model = AutoModelForSeq2SeqLM.from_pretrained(sys.path[1] + 'models/best.pt')
model.eval()
model.config.use_cache = False


def translate(model, inference_request, tokenizer=tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0],
          skip_special_tokens=True, temperature=0))


sent = test_df.sample(n=5).source

for s in sent:
    inference_request = prefix + s
    print(s)
    print(translate(model, inference_request, tokenizer))
    print()
