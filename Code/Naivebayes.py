from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv('data_label.csv', index_col=[0], on_bad_lines='skip')
tweet_df = df[['tweet', 'sentiment']]

# Split into training and testing data
x = tweet_df['tweet']
y = tweet_df['sentiment']
x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state= 22)


# Vectorize text reviews to numbers
txt_vector = CountVectorizer(stop_words='english')
x = txt_vector.fit_transform(x).toarray()
x_test = txt_vector.transform(x_test).toarray()

model = MultinomialNB()
print('CS5344 Project: Group 22')
print('Running naivebayes now! Please wait for a while......')
model.fit(x, y)
score = model.score(x_test, y_test)
print(f'Naive Bayes Accuracy: {score}%')
# 0.7629134299671658
