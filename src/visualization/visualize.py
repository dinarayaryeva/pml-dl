from wordcloud import WordCloud, STOPWORDS
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt
import sys
import pandas as pd
import sys


def uncommon_words(A, B):
    count = {}
    for word in A.split():
        count[word] = count.get(word, 0) + 1

    for word in B.split():
        count[word] = count.get(word, 0) + 1

    return [word for word in count if count[word] == 1]


def translate(model, inference_request, tokenizer):
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0],
                            skip_special_tokens=True, temperature=0)


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

N = 1000
sent = test_df.sample(n=N).source.values.tolist()
trn = []
for s in sent:
    inference_request = prefix + s
    t = translate(model, inference_request, tokenizer)
    trn.append(t)

uncommon_words_counter = {}
meeted_words = []
for i in range(N):
    words = uncommon_words(sent[i], trn[i])
    for wrd in words:
        if len(wrd) > 3:
            if wrd in meeted_words:
                uncommon_words_counter[wrd] = uncommon_words_counter[wrd] + 1
            else:
                uncommon_words_counter[wrd] = 1
                meeted_words.append(wrd)

# Create stopword list:
stopwords = set(STOPWORDS)

# Generate a word cloud image
wordcloud = WordCloud(stopwords=stopwords).generate_from_frequencies(
    uncommon_words_counter)

# Display the generated image:
# the matplotlib way:
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
