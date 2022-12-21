import pandas as pd
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SpatialDropout1D
from tensorflow.keras.layers import Embedding
import numpy as np
import pandas as pd

df = pd.read_csv('data_label.csv', index_col=[0], on_bad_lines='skip')

tweet_df = df[['tweet', 'sentiment']]
sentiment_label = tweet_df.sentiment.factorize()


#Read tweet content from preprocessed data
tweet = tweet_df.tweet.values

#Set word limit 5000
tokenizer = Tokenizer(num_words=5000)

#Fit on tokenizer
tokenizer.fit_on_texts(tweet)

vocab_size = len(tokenizer.word_index) + 1

#Encode tweets into sequences
sequence = tokenizer.texts_to_sequences(tweet)
padded_sequence = pad_sequences(sequence, maxlen=200)


model = Sequential()
model.add(Embedding(vocab_size, 32, input_length=200))
model.add(SpatialDropout1D(0.25))
model.add(LSTM(50, dropout=0.5, recurrent_dropout=0.5))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print('CS5344 Project: Group 22')
print('Running LSTM now! Please wait for a while......')
print(model.summary())
history = model.fit(padded_sequence, sentiment_label[0], validation_split=0.2, epochs = 5, batch_size=32)
loss, acc = model.evaluate(padded_sequence, sentiment_label[0])
print(f'LSTM Accuracy: {(100*acc)}%')




