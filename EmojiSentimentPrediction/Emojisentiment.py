import emoji
import nltk
import numpy
import sklearn
import pandas
from nltk.tokenize import TweetTokenizer
from collections import Counter

# read data: with sentimens(twitter_data) & whithout(emojitweets) & merge them.
twitter_sentiment = pandas.read_csv("Twitter_Data.csv", encoding='utf8', nrows=50000)   
emojitweets = pandas.read_csv('emojitweets_10000.txt', sep="\n", header=None, nrows=5000)
emojitweets.columns = ["clean_text"]
df = pandas.concat([emojitweets, twitter_sentiment])
df['clean_text'] = df['clean_text'].apply(repr)

# A. TRAIN MODEL WITH TWO FEATURE EXTRACTION METHODS: BOW AND TFIDF
## A.1. BOW CLASSIFICATION
# vectorize text with bag of words: 
vectorizer = sklearn.feature_extraction.text.CountVectorizer(token_pattern=r'[^\s]+')
vectorizer.fit(df['clean_text'])
x_train = vectorizer.transform(df['clean_text'][5000:])
y_train = df['category'][5000:]
# train model with twitter_sentiment:
model = sklearn.linear_model.LogisticRegression(max_iter=500)
model.fit(x_train, y_train)
# classify emojitweets with trained model:
x_new = vectorizer.transform(df['clean_text'][:4000])
y_new = model.predict(x_new)
df_2 = pandas.DataFrame()
df_2['category'] = y_new
df_2['clean_text'] = df['clean_text'][:4000]
df = pandas.concat([df_2, df[4000:]])

# # A.2. TF-IDF
# # vectorize text with tfidf and train model:
# vectorizer_1 = sklearn.feature_extraction.text.TfidfVectorizer()
# vectorizer_1.fit(df['clean_text'])
# x_train_1 = vectorizer_1.transform(df['clean_text'][5000:])
# model.fit(x_train_1, y_train)
# # classify emojitweets with trained model
# x_new_1 = vectorizer_1.transform(df['clean_text'][:4000])
# y_new_1 = (model.predict(x_new_1))
# df_2 = pandas.DataFrame()
# df_2['category'] = y_new_1
# df_2['clean_text'] = df['clean_text'][:4000]
# df = pandas.concat([df_2, df[4000:]])
    
### B. Compare sentiment from text with emojis and without them: 
# extract emojis from data:
tk =TweetTokenizer()

def remove_emojis (text):
    "removes emojis from text"
    help_text = []
    for word in text:
        if word not in emoji.UNICODE_EMOJI:
            help_text.append(word)
    return(help_text)

emojitweets['tokenized'] = emojitweets['clean_text'].apply(tk.tokenize)
df_1 = pandas.DataFrame()
df_1["clean_text"] = emojitweets['tokenized'].apply(remove_emojis)

## detokenize text:to do the prediction we need to detokenize everything
df_1["clean_text"] = df_1["clean_text"].apply(
    nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize)

## add all data (tweets with extracted emojis, tweets with emojis and predicted 
##               sentiment and tweets with sentiment to one dataframe)
df = pandas.concat([df[5000:], df[:4000], df[4000:5000], df_1[4000:]])
"""
the resulting df consists on: 
    -50000 originalpredicted tweets without emojis
    -5000 tweets with emojis and predicted sentiments
    -1000 tweets with emojis (not predicted)
    -1000 the same 1000 tweets as right above but with extracted emojis

"""
df['clean_text'] = df['clean_text'].apply(repr)   
   
# train model with data: emojitweets+predicted sentiment and tweets with sentiment:
vectorizer_2 = sklearn.feature_extraction.text.TfidfVectorizer(token_pattern=r'[^\s]+')
vectorizer_2.fit(df['clean_text'][:55000])
x_train_2 = vectorizer_2.transform(df['clean_text'][50000:54000])
y_train_2 = df['category'][50000:54000]
model.fit(x_train_2, y_train_2)

# predict sentiment for tweets with extracted emojis:
x_new_2 = vectorizer_2.transform(df['clean_text'][55000:])
y_new_2 = model.predict(x_new_2)
# predict sentiment for tweets with emojis: 
x_new_3 = vectorizer_2.transform(df['clean_text'][54000:55000])
y_new_3 = model.predict(x_new_3)

# compare sentiments of tweets with emojis and with extracted emojis :
y_new_2_count = Counter(y_new_2)
print('sentiments of tweets without emojis:', y_new_2_count)
y_new_3_count = Counter(y_new_3)
print('sentiments of tweets with emojis:', y_new_3_count)



    


