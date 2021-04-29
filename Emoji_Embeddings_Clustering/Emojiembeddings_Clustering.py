# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 18:09:24 2020
@author: Victor, David and Lucia
"""

import emoji
import nltk
import numpy
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

'''GLOBAL VARIABLES AND INITS'''
pd.set_option('display.max_columns', 99)
PATH = "emoji_df.csv"
DATA_SIZE = 400
df = pd.read_csv(PATH, nrows=DATA_SIZE)
rank = pd.read_csv("ijstable.csv", nrows = 30)
tk = TweetTokenizer()
sentence = []
emojitweets = []
dict_emoji = {}
cluster_aux_data = pd.DataFrame()

def norm(vector):
    return numpy.sqrt(sum([a*a for a in vector]))
def dot(vector1, vector2):
    return sum([i * j for i, j in zip(vector1, vector2)])
def cosine(vector1, vector2):    
    ret=dot(vector1, vector2) / (norm(vector1) * norm(vector2))
    return ret

def take_vector(string):
    '''
    takes a string as input to return it as a vector.
    If the input is an emoji as "image" it directly searches for its vector,
    but in case it is a "written emoji" (such as ":thumbs_up:") it first 
    tranformes it to the image version before making the vectorization.
    '''
    if ':' in string:
        string=emoji.emojize(string)
    return(model_emojis.wv.__getitem__(string))

def take_best_emoji(vector):
    '''
    Takes the vectorized sentence or word as input and searches
    in the dictionary of emojis (created with the corpus) the most
    fitting emoji. It allows to return more than one emoji if the
    difference in the cosine between the emojis and the word is equal
    or less than 0.02
    '''
    cos=-1
    em=0
    for emo in dict_emoji:
        c=cosine(vector,dict_emoji[emo])
        if c>(cos+0.02):
            cos=c
            em=emo
        elif c>=cos or c>=(cos-0.02):
            if type(em)!=list:
                em=[em]
                em.append(emo)
            else:
                em.append(emo)
    return em


def model_sentences(sentence):
    '''  Takes a sentence (string) and returns it as a vector '''
    tokenized_sentence=nltk.tokenize.word_tokenize(sentence)
    words_to_be_used = [word for word in tokenized_sentence if model_emojis.wv.__contains__(word)]
    word_embs = [model_emojis.wv.__getitem__(word) for word in words_to_be_used]
    if not word_embs:
        return numpy.random.normal(size =(300,))
    return numpy.mean(word_embs, axis = 0)


def predict_emojis_sentence(sentence):
    '''
    By using the previous functions model_sentence and take_best_emoji
    returns the emoji (or emojis) that fit the best that sentence
    '''
    sentence=model_sentences(sentence)
    em=take_best_emoji(sentence)
    print()
    return em

def predict_emojy_by_input():
    ''' Es una pequeña tontería que he hecho para que se le pueda escribir 
    frases todo el rato y fácilmente. Para pararlo sólo tienes que escribir 0 
    '''
    a='1'
    while a != '0':
        a=input('Give me your sentence: ')
        print(a, predict_emojis_sentence(a))
        
''' Method to reduce the emojis to be plotted '''
def getNRandomSamples(list1, list2, list3, cl, N):
    new_list1 = []
    new_list2 = []
    new_list3 = []
    labels_cl = []
    index = 0
    for x in range(N):
        if index > len(list1): 
            index = 0
        else:
            new_list1.append(list1[index])
            new_list2.append(list2[index])
            new_list3.append(list3[index])
            labels_cl.append(cl[index])
            a = random.randint(1,7)
            index = index + a
    return {'x_ax':new_list1, 'y_ax':new_list2, 
            'labels_t':new_list3, 'labels_c':labels_cl}     

'''*************************** CLUSTERING METHOD ***************************'''
def cluster_emojis():
    ''' Set codepint in right format U+12345 '''
    df['emojipedia_codepoint'] = 'U+' + df['codepoints']    
    print('Start clustering part...')
    
    ''' Ponemos los embeddings en una clumna de pandas '''
    embeddings = []
    for i, row in df.iterrows():
        emoji = row['emoji']
        if emoji in dict_emoji:
            emb = dict_emoji[emoji]
        else: 
            emb = numpy.zeros(shape =(100,))
            
        embeddings.append(emb)
            
    df['embedding'] = embeddings
    
    ''' Reduced dataframe with all necesssary info '''
    cluster_df = df[['emoji','sub_group','emojipedia_codepoint','embedding']]
    
    NUM_CLUSTERS=6
    
    ''' Get needed lists '''
    X = list(cluster_df['embedding'])
    codes = list(cluster_df['emojipedia_codepoint'])
    # emojis = list(cluster_df['emoji'])
    # groups = list(cluster_df['sub_group'])
    # names= list(cluster_df['emoji_name'])
    
    ''' Reduce dimensionality for cluster plotting '''
    model = TSNE(n_components=2, random_state=0)
    numpy.set_printoptions(suppress=True)
    Y=model.fit_transform(X)
    
    data = [[], []]
    [data[0].append(i) for i,j in Y]
    [data[1].append(j) for i,j in Y]
    
    
    ''' Calcular los clusters'''
    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(Y)
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    colors = ["r", "b", "c", "y", "m", "g" ]
    
    '''Get only N emojis randomly'''
    cont = getNRandomSamples(data[0], data[1], codes, labels, 30)
    x_ax = cont['x_ax']
    y_ax = cont['y_ax']
    label_text = cont['labels_t']
    label_cluster = cont['labels_c'] 
    
    '''Create DataFrame to support cluster plots'''
    cluster_aux_data['code'] = label_text
    cluster_aux_data['cluster'] = label_cluster
    cluster_aux_data['color'] = cluster_aux_data.apply(lambda x: colors[x['cluster']], axis =1)
    
    fig, ax = plt.subplots(figsize=(20,20))
    ax.scatter(x_ax, y_ax, c=[colors[d] for d in label_cluster])
    ''' Add annotations (code names) to the plot'''
    for i, txt in enumerate(label_text):
        ax.annotate(txt, 
                    (x_ax[i], y_ax[i]),
                    xytext = (-20,20), 
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', 
                                    connectionstyle='arc3,rad=0'),
                    fontname='Segoe UI Emoji', 
                    fontsize=20)

    '''Plot centroids'''
    ax.plot(kmeans.cluster_centers_[0][0], kmeans.cluster_centers_[0][1], 
            'ro',markersize=36, alpha = 0.25, label='')
    ax.plot(kmeans.cluster_centers_[1][0], kmeans.cluster_centers_[1][1], 
            'bo',markersize=36, alpha = 0.25)
    ax.plot(kmeans.cluster_centers_[2][0], kmeans.cluster_centers_[2][1], 
            'co',markersize=36, alpha = 0.25)
    ax.plot(kmeans.cluster_centers_[3][0], kmeans.cluster_centers_[3][1], 
            'yo',markersize=36, alpha = 0.25)
    ax.plot(kmeans.cluster_centers_[4][0], kmeans.cluster_centers_[4][1], 
            'mo',markersize=36, alpha = 0.25)
    ax.plot(kmeans.cluster_centers_[5][0], kmeans.cluster_centers_[5][1], 
            'go',markersize=36, alpha = 0.25)
      
    print ("Cluster id labels for inputted data")
    print (labels)
    print ("Centroids data")
    print (centroids)
    print ("Score: ", kmeans.score(Y))
    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    print ("Silhouette_score: ", silhouette_score)


with open("emojitweets.txt", encoding='utf8') as file:
    for x in file:
        sentence.append(x)
        
''' Get the 1.000.000 sentences '''
for x in sentence[:1000000]:
    emojitweets.append(tk.tokenize(x.rstrip()))       

model_emojis = Word2Vec(emojitweets)
''' Create the emojis dictionary with its embedding '''
for sentence in emojitweets:
    for word in sentence:
        if word in emoji.UNICODE_EMOJI and word not in dict_emoji and model_emojis.wv.__contains__(word):
            dict_emoji[word]=take_vector(word)
    
# If you want to test the emoji prediction for sentences, toggle play() comment
# predict_emojy_by_input()  
cluster_emojis()


 