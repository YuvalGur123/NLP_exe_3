import numpy as np
import nltk
from nltk import pos_tag, RegexpTokenizer
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense, LSTM
from tensorflow.keras.utils import to_categorical
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.kl import KLSummarizer
from transformers import GPT2Tokenizer, TFGPT2LMHeadModel, TFTrainer, TFTrainingArguments, \
    DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
import torch

### text processing from exe 1, 2 ####

f = open("example_text.txt")
string_text = f.read()

sentence_tokens = nltk.sent_tokenize(string_text)
word_tokenizer = RegexpTokenizer(r'\w+|[^\w\s]')

tokens_list = [nltk.word_tokenize(sentence) for sentence in sentence_tokens]
tokens_flat = [token for sentence_tokens in tokens_list for token in sentence_tokens]

tokens = word_tokenizer.tokenize(' '.join(tokens_flat))

tokens_lower = [x.lower() for x in tokens]
lemmatizer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(i, j[0].lower()) if j[0].lower() in ['a', 'n', 'v'] else lemmatizer.lemmatize(i) for i, j in pos_tag(tokens_lower)]

ps = PorterStemmer()
stemmed_text = [ps.stem(token) for token in tokens_lower]

stop_words = set(stopwords.words('english'))

corpus = [token for token in lemmas if token not in stop_words]

tokenizer = {}
reverse_tokenizer = {}
index = 1
for word in corpus:
    if word not in tokenizer:
        tokenizer[word] = index
        reverse_tokenizer[index] = word
        index += 1

###### 2) applying RNN to predict words

# Convert corpus into sequences of indices
sequences = []
for i in range(1, len(corpus)):
    n_gram_sequence = corpus[:i+1]
    sequences.append([tokenizer[word] for word in n_gram_sequence])

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
X, y = sequences[:, :-1], sequences[:, -1]
y = to_categorical(y, num_classes=len(tokenizer)+1)


# Build the RNN model

model = Sequential()
model.add(Embedding(len(tokenizer)+1, 50, input_length=X.shape[1]))
model.add(SimpleRNN(100))
model.add(Dense(len(tokenizer)+1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_RNN = model.fit(X, y, epochs=1, verbose=1)


# Predict the next word
def predict_next_word(model, tokenizer, reverse_tokenizer, text):
    sequence = [tokenizer.get(word, 0) for word in text.split()]
    sequence = pad_sequences([sequence], maxlen=X.shape[1], padding='pre')
    predicted_probs = model.predict(sequence, verbose=0)
    predicted_word_index = np.argmax(predicted_probs)
    return reverse_tokenizer.get(predicted_word_index, '')


text = 'what types of'
next_word = predict_next_word(model, tokenizer, reverse_tokenizer, text)
print(f'Next word prediction RNN: {next_word}')

print("History rnn", history_RNN.history.keys())


#### 3) using LSTM to predict words #####

# Build the LSTM model

model = Sequential()
model.add(Embedding(len(tokenizer)+1, 50, input_length=X.shape[1]))
model.add(LSTM(100, return_sequences=False))  # Use LSTM instead of SimpleRNN
model.add(Dense(len(tokenizer)+1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_LSTM = model.fit(X, y, epochs=1, verbose=1)


text = 'what types of'
next_word = predict_next_word(model, tokenizer, reverse_tokenizer, text)
print(f'Next word prediction LSTM: {next_word}')

#### 4) comparing accuracy and perplexity of models

perplexity_RNN = np.exp(history_RNN.history['loss'][-1])
perplexity_LSTM = np.exp(history_LSTM.history['loss'][-1])

print("RNN perplexity: ", perplexity_RNN)
print("LSTM perplexity: ", perplexity_LSTM)

accuracy_RNN = history_RNN.history['accuracy'][-1]
accuracy_LSTM = history_LSTM.history['accuracy'][-1]

print("RNN accuracy: ", accuracy_RNN)
print("LSTM accuracy: ", accuracy_LSTM)

corpus2 = [''.join(tokens) for tokens in corpus]
text2 = ' '.join(corpus2)


#### 5) using KL-sum to summarize the corpus #####

parser = PlaintextParser.from_string(text2, Tokenizer("english"))
summarizer = KLSummarizer()
summary = summarizer(parser.document, 3)
for sentence in summary:
    print(sentence)

text_tokens = " ".join(tokens_lower)

##### 6) fine-tuning GPT2 model ####

## the commented out pard is the train part, it only need to be ran once to create the model

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
# dataset = load_dataset('text', data_files={'train': 'corpus.txt'})
#
#
# def tokenize_function(examples):
#     tokenized_inputs = tokenizer(examples['text'], padding='max_length', truncation=True)
#
#     # Set the labels equal to the input IDs
#     tokenized_inputs['labels'] = tokenized_inputs['input_ids'].copy()
#
#     # Replace padding token labels with -100 to ignore them in loss calculation
#     tokenized_inputs['labels'] = [
#         [(label if label != tokenizer.pad_token_id else -100) for label in labels]
#         for labels in tokenized_inputs['labels']
#     ]
#
#     return tokenized_inputs
#
#
# tokenized_datasets = dataset.map(tokenize_function, batched=True)
# model = GPT2LMHeadModel.from_pretrained('gpt2')
# model.resize_token_embeddings(len(tokenizer))
#
# training_args = TrainingArguments(
#     output_dir='./results',          # output directory
#     evaluation_strategy="no",     # evaluation strategy
#     learning_rate=2e-5,              # learning rate
#     per_device_train_batch_size=2,   # batch size
#     per_device_eval_batch_size=2,    # eval batch size
#     num_train_epochs=3,              # number of epochs
#     weight_decay=0.01,               # strength of weight decay
# )
# trainer = Trainer(
#     model=model,                         # the instantiated GPT-2 model
#     args=training_args,                  # training arguments
#     train_dataset=tokenized_datasets['train'],    # training dataset
# )
# trainer.train()
# model.save_pretrained("fine-tuned-gpt2")
# tokenizer.save_pretrained("fine-tuned-gpt2")

#### actual use of the trained model

model_name = "fine-tuned-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

## Complete sentences:
# 1: tokenization is a critical step in many nlp tasks
# 2: tokenization reduces the size of raw text so that it can be handled more easily
# 3: tokenization is the process of dividing a text into smaller units known as tokens.
# 4: language modelling: tokenization in nlp facilitates the creation of organized representations of language
# 5: this rapid conversion enables the immediate utilization of these tokenized elements by a computer to initiate practical actions and responses.
incomplete_sentences = [
    "tokenization is a critical step in",
    "tokenization reduces the size of raw",
    "tokenization is the process of dividing a text into smaller",
    "language modelling: tokenization in nlp facilitates",
    "this rapid conversion enables the immediate utilization"
]

for i, sentence in enumerate(incomplete_sentences):
    # Tokenize the input sentence
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(device)

    # Generate predictions using the model
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=input_ids.size(-1) + 15,  # Maximum length of the generated sequence
            num_return_sequences=1,  # Number of sequences to generate
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
            early_stopping=True,
            eos_token_id=tokenizer.eos_token_id,  # Stop when EOS token is generated
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the output to get the predicted sentence
    predicted_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Print the original and predicted sentences
    print(f"Original Sentence {i+1}: {sentence}")
    print(f"Predicted Sentence {i+1}: {predicted_text}\n")

# Original Sentence 1: tokenization is a critical step in
# Predicted Sentence 1: tokenization is a critical step in the development of a language.
#
# Original Sentence 2: tokenization reduces the size of raw
# Predicted Sentence 2: tokenization reduces the size of raw data and reduces processing time.
#
# Original Sentence 3: tokenization is the process of dividing a text into smaller
# Predicted Sentence 3: tokenization is the process of dividing a text into smaller chunks of text. The process is called tokenization.
#
# Original Sentence 4: language modelling: tokenization in nlp facilitates
# Predicted Sentence 4: language modelling: tokenization in nlp facilitates the identification of words and phrases in a language.
#
# Original Sentence 5: this rapid conversion enables the immediate utilization
# Predicted Sentence 5: this rapid conversion enables the immediate utilization of a large number of data sets in a single process.


##### 7) sentiment analysis