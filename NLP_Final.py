# -*- coding: utf-8 -*-
"""
Created on Sun May 15 15:12:00 2020

@author: Arunima
"""

import re
import pickle
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from functools import partialmethod
pd.DataFrame.head = partialmethod(pd.DataFrame.head, n=2)
from gensim.corpora import Dictionary
from sklearn import preprocessing
from collections import Counter
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.stem import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup
import html as ihtml
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, plot_roc_curve, classification_report

# 18 fields expected.
field_names = ['id', 'type', 'is_best_answer', 'topic_id', 'parent_id', 'votes',
               'title', 'content', 'member', 'category', 'state', 'is_solved', 'num_answers',
               'country', 'date', 'last_answer_date', 'author_crc', 'visits']


txt_path = "C:/Users/Arunima/Downloads/export-forums_en.csv/export-forums_en.csv"
entity_path = "C:/Users/Arunima/Downloads/export-forums_en.csv/export-forums_en.pickle"
csv_path = "C:/Users/Arunima/Downloads/export-forums_en.csv/export-forums_en.format.csv"
data_path = "C:/Users/Arunima/Downloads/export-forums_en.csv/"

def format_entities():
    '''
    Read the raw data, format the list of entities, serialize them.
    '''

    def build_entities(txt_path, max_entities=None):
        '''
        Return a list of structured entities from raw txt file.
        '''
        # Read text file.
        with open(txt_path, 'r', encoding='utf8') as f:
            # Entities and current entity.
            entities, entity = [], {}
            # Entity values might be split over lines
            field_counter = 0
            # Process lines
            for line in f:
                # Change value
                line = line.replace("\\N", '"unkwown"')
                # Char start for extracted value.
                char_start = 1
                # Find values separators
                field_index = [m.start() for m in re.finditer('","', line)]
                # Browse value separators.
                for index in field_index:
                    # Extract in between value.
                    value = line[char_start:index]
                    # Update start index.
                    char_start = index + 3
                    # Update field counter.
                    field_counter += 1
                    # Update entity value.
                    try:
                        entity[field_names[field_counter-1]] += value
                    except KeyError:
                        entity[field_names[field_counter-1]] = value
                    except IndexError:
                        entity = {}
                        field_counter = 0
                # Content string is split.
                if field_counter == 7 and len(field_index) > 0:
                    entity[field_names[7]] = line[field_index[-1]:]
                    continue
                # Next content string.
                if field_counter == 7 and len(field_index) == 0:
                    entity[field_names[7]] += line
                    continue
                # Next entity.
                if len(entity) == 17:
                    field_counter = 0
                    entities.append(entity)
                    entity = {}
                    if max_entities is not None:
                        if len(entities) > max_entities:
                            return entities
        return entities

    # Write entities on disk.
    with open(entity_path, 'wb') as f:
        pickle.dump(build_entities(txt_path=txt_path, max_entities=None), f)
        
format_entities()

#Data Exploration

with open(entity_path, 'rb') as obj:
        entities = pickle.load(obj)
df = pd.DataFrame(entities)
print(df.shape)
df.head()

table1 = []
for col in df.columns:
    table1.append((col, df[col].nunique(), df[col].isnull().sum(),  df[col].dtype))
    
table1_df = pd.DataFrame(table1, columns=['Variable Name', 'Unique Value', 'Missing Value', 'Data Type'])
table1_df

# convert data types
num_cols = ['is_best_answer','topic_id','parent_id','votes','member','state',
              'is_solved','num_answers','date','last_answer_date','author_crc']
df[num_cols] = df[num_cols].apply(pd.to_numeric)
df.info()

df['type'].unique()

# separate date by Question and Answer
Q = df[df['type']=='Q']
A = df[df['type']=='A']
C = df[df['type']=='C']

# Check if there's any questions without answer
print("Questions without answers:", round(len(Q[Q['num_answers'] == 0])/len(Q)*100),'%')

#Data Cleaning
df.drop(['id','topic_id', 'parent_id', 'title', 'author_crc'], axis=1, inplace=True)

# Calculate the duration from active date
df['duration'] = df['last_answer_date'] - df['date']

# Drop all the date columns except durations
df.drop(['date','last_answer_date'], axis=1, inplace=True)
df.head()

#Encoding
# Label Encoding
encoder = preprocessing.LabelEncoder()
df['category'] = encoder.fit_transform(df['category'])
df['country'] = encoder.fit_transform(df['country'])
# OneHot Encoding
df['type'] = pd.get_dummies(df['type'], drop_first=True)

#Cleaning Text
def clean_sentences(sentences):
    tokens = []
    
    for sentence in sentences:
        text = re.sub('http.*', '', sentence)
        soup = BeautifulSoup(text, 'lxml')
        a = soup.get_text()
        t = a.strip()
        token = word_tokenize(t)
        token = [w.lower() for w in token]
        table = str.maketrans('', '', string.punctuation)
        stripped = [w.translate(table) for w in token]
        words = [word for word in stripped if word.isalpha()]
        stop_words = set(stopwords.words('english'))
        words = [w for w in words if not w in stop_words]
        tokens.append(words)

    return tokens

df['content'] = clean_sentences(df['content'])

df.to_csv('df_cleaned.csv', index=False)

#Embedding : Using Word2Vec
df = pd.read_csv('df_cleaned.csv')
df.shape

model_w2v = Word2Vec(df['content'], min_count = 1,size = 50,workers = 3, window = 3, sg = 1)
model_w2v[model_w2v.wv.vocab]

def build_word2vec_from_text(model_w2v, sentence, emb_size):
    emb_vec = np.zeros(emb_size).reshape((1, emb_size))
    count = 0.
    for word in sentence:
        try:
            emb_vec += model_w2v[word].reshape((1, emb_size))
            count += 1.
        except KeyError:
            continue
    if count != 0:
        emb_vec /= count
    return emb_vec

df['content'] = np.concatenate([build_word2vec_from_text(model_w2v, d, 50) for d in df['content']], axis=0)

#Resample the data
best = df['is_best_answer'].values
y = np.array(best.copy())
X = df.drop(columns=['is_best_answer']).values
y = (y > 0)*1

nb_class = len(set(y))
nb_class

# create empty arrays for both classes
id_Train = np.array([None]*nb_class)
id_Test = np.array([None]*nb_class)

for i in range(nb_class):
    id_i = np.where(y==i)[0]
    id_i_train, id_i_test = train_test_split(id_i, test_size=0.3)
    id_Train[i] = id_i_train
    id_Test[i] = id_i_test
    
id_Train = np.concatenate(id_Train)
id_Test = np.concatenate(id_Test)

X_train = X[id_Train]
X_test = X[id_Test]
y_train = y[id_Train]
y_test = y[id_Test]

id_toTrain = np.array([np.where(y_train==i)[0] for i in range(nb_class)])

size_max = [len(id_toTrain[i]) for i in range(nb_class)]
print("Before Resampling", size_max)

blc = 800
for i in range(len(size_max)):
    if size_max[i] > blc:
        size_max[i] = int(blc*(np.log10(size_max[i]/blc)+100))
    else:
        size_max[i] = int(blc/(np.log10(blc/size_max[i])+0.05))
        
print("After Resampling", size_max)

for i in range(nb_class):
    if len(id_toTrain[i]) > size_max[i]:
        id_toTrain[i], tmp = train_test_split(id_toTrain[i], test_size=1-size_max[i]/len(id_toTrain[i]))
    else:
        id_toTrain[i] = np.concatenate((id_toTrain[i], id_toTrain[i][np.random.randint(len(id_toTrain[i]), size=int(size_max[i]-len(id_toTrain[i])))]))
id_toTrain = np.concatenate(id_toTrain)
X_train = X_train[id_toTrain]
y_train = y_train[id_toTrain]

print(Counter(y_train))

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)
X_train.columns = ['type', 'votes', 'content', 'member', 'category', 'state', 'is_solved', 'num_answers', 'country', 'duration']
X_test.columns = ['type', 'votes', 'content', 'member', 'category', 'state', 'is_solved', 'num_answers', 'country', 'duration']

#Modeling using XGBoost
xgb = XGBClassifier(
    learning_rate=0.1, max_depth=5,
    min_child_weight=10, min_samples_leaf=5, min_samples_split=5,
    n_estimators=50, objective='binary:logistic')

xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_pred_prob_xgb = xgb.predict_proba(X_test)[:,0]

print("ROC-AUC", roc_auc_score(y_test, y_pred_xgb))
print("Confusion Matrix")
print(classification_report(y_test, y_pred_xgb))

fig, ax = plt.subplots(figsize=(10, 8))
ax = plot_importance(xgb, importance_type = 'weight', max_num_features=10, ax=ax)
plt.rc('font', size=15)
plt.show()

eval_set = [(X_train, y_train), (X_test, y_test)]
xgb.fit(X_train, y_train, eval_metric=["error","logloss"], eval_set=eval_set, verbose=0)
results = xgb.evals_result()
epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
plt.rc('font', size=15)
plt.show()

# plot classification error
fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.rc('font', size=15)
plt.show()

# plot ROC-AUC curve
plt.subplots(1, 1, figsize=(10, 8))
ax = plt.subplot(1,1,1)
plt.title('ROC-AUC Curve')
plt.rc('font', size=15)
plot_roc_curve(xgb, X_test, y_test, ax=ax)
