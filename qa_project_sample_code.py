# my thanks go to Baptiste + DP
# A-P
import os
import numpy as np
from scipy.sparse import csc_matrix, save_npz, load_npz
import re
import pickle
import pandas as pd
from datetime import datetime
from gensim.corpora import Dictionary
from nltk.tokenize import TweetTokenizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from nltk.stem import PorterStemmer
#import matplotlib.pyplot as plt

# 18 fields expected.
field_names = ['id', 'type', 'is_best_answer', 'topic_id', 'parent_id', 'votes',
               'title', 'content', 'member', 'category', 'state', 'is_solved', 'num_answers',
               'country', 'date', 'last_answer_date', 'author_crc', 'visits']


txt_path = "../QA/export-forums_en.csv"
entity_path = "../QA/export-forums_en.pickle"
csv_path = "../QA/export-forums_en.format.csv"
data_path = "../QA/"


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
                # Prepare line
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


def entities_to_csv():
    '''
    Format entities to csv.
    '''
    with open(entity_path, 'rb') as obj:
        entities = pickle.load(obj)
    x = pd.DataFrame(entities)
    x.to_csv(csv_path)


def build_index():
    '''
    Build index and write.
    '''

    # Read entities.
    with open(entity_path, 'rb') as obj:
        entities = pickle.load(obj)

    # Build indexes
    user_index, question_index, answer_index, comment_index = {}, {}, {}, {}
    for e in entities:
        # Author
        if e['author_crc'] not in user_index:
            user_index[e['author_crc']] = len(user_index)
        # Questions
        if e['type'] == 'Q':
            if e['id'] not in question_index:
                question_index[e['id']] = len(question_index)
        # Answers
        if e['type'] == 'A':
            if e['id'] not in answer_index:
                answer_index[e['id']] = len(answer_index)
        # Answers
        if e['type'] == 'C':
            if e['id'] not in comment_index:
                comment_index[e['id']] = len(comment_index)

    # Write index.
    with open(os.path.join(data_path, 'user.index'), 'wb') as f:
        pickle.dump(user_index, f)
    with open(os.path.join(data_path, 'question.index'), 'wb') as f:
        pickle.dump(question_index, f)
    with open(os.path.join(data_path, 'answer.index'), 'wb') as f:
        pickle.dump(answer_index, f)
    with open(os.path.join(data_path, 'comment.index'), 'wb') as f:
        pickle.dump(comment_index, f)

    # Logs.
    print("Entities: ", len(entities))
    print("Users: ", len(user_index))
    print("Questions: ", len(question_index))
    print("Answers: ", len(answer_index))
    print("Comments: ", len(comment_index))


def read_indexes():
    '''
    Return user, question, answer and comment index.
    '''
    with open(os.path.join(data_path, 'user.index'), 'rb') as obj:
        user_index = pickle.load(obj)
    with open(os.path.join(data_path, 'question.index'), 'rb') as obj:
        question_index = pickle.load(obj)
    with open(os.path.join(data_path, 'answer.index'), 'rb') as obj:
        answer_index = pickle.load(obj)
    with open(os.path.join(data_path, 'comment.index'), 'rb') as obj:
        comment_index = pickle.load(obj)
    return user_index, question_index, answer_index, comment_index


def build_relations():
    '''
    Build UQ, UA, QA, UC, CA relations.
    '''

    # Read entities.
    with open(entity_path, 'rb') as obj:
        entities = pickle.load(obj)

    # Read indexes
    user_index, question_index, answer_index, comment_index = read_indexes()

    # Relations
    uq = []
    ua = []
    qa = []
    uc = []
    ca = []

    # Browse elements.
    for e in entities:

        # UQ
        if e['type'] == 'Q':
            u = user_index[e['author_crc']]
            q = question_index[e['id']]
            t = datetime.utcfromtimestamp(int(e['date'])).strftime('%Y-%m-%d %H:%M:%S')
            uq.append((u,q,t))

        # UA, QA
        if e['type'] == 'A':
            u = user_index[e['author_crc']]
            a = answer_index[e['id']]
            q = question_index[e['parent_id']]
            t = datetime.utcfromtimestamp(int(e['date'])).strftime('%Y-%m-%d %H:%M:%S')
            ua.append((u,a,t))
            qa.append((q,a,t))

        # UC, CA
        if e['type'] == 'C':
            try:
                u = user_index[e['author_crc']]
                c = comment_index[e['id']]
                a = answer_index[e['parent_id']]
                t = datetime.utcfromtimestamp(int(e['date'])).strftime('%Y-%m-%d %H:%M:%S')
                uc.append((u,c,t))
                ca.append((c,a,t))
            except KeyError:
                continue

    # Write relations.
    with open(os.path.join(data_path, 'uq.rel'), 'wb') as f:
        pickle.dump(uq, f)
    # Write relations.
    with open(os.path.join(data_path, 'ua.rel'), 'wb') as f:
        pickle.dump(ua, f)
    # Write relations.
    with open(os.path.join(data_path, 'qa.rel'), 'wb') as f:
        pickle.dump(qa, f)
    # Write relations.
    with open(os.path.join(data_path, 'uc.rel'), 'wb') as f:
        pickle.dump(uc, f)
    # Write relations.
    with open(os.path.join(data_path, 'ca.rel'), 'wb') as f:
        pickle.dump(ca, f)

    # Logs.
    print("uq: ", len(uq))
    print("ua: ", len(ua))
    print("qa: ", len(qa))
    print("uc: ", len(uc))
    print("ca: ", len(ca))


def build_vocabulary_and_corpus():
    '''
    Build the vocabularies and stem sequences for each type of entities.
    '''

    # Vocabulary (same for question and answers)
    v = Dictionary()

    # Stemmer.
    stemmer = PorterStemmer()

    # Tokenizer.
    tokenizer = TweetTokenizer()

    # Read indexes
    user_index, question_index, answer_index, comment_index = read_indexes()

    # Question, answer
    q = {}
    a = {}

    # Read entities.
    with open(entity_path, 'rb') as obj:
        entities = pickle.load(obj)

    # Browse question and answers to first build vocabulary.
    for e in entities:
        # Question or answer.
        if e['type'] == 'Q' or e['type'] == 'A':
            # String content.
            title = str(e['title']).encode('utf-8').lower()
            content = str(e['content']).encode('utf-8').lower()
            # Tokenize
            d = tokenizer.tokenize(title + content)
            # Stem word
            d = [stemmer.stem(s) for s in d]
            # Process vocabulary.
            v.add_documents([d])
            # Question
            if e['type'] == 'Q':
                q[question_index[e['id']]] = d
            # Answer
            if e['type'] == 'A':
                a[answer_index[e['id']]] = d

    # Write question corpus.
    with open(os.path.join(data_path, 'q.corpus'), 'wb') as f:
        pickle.dump(q, f)

    # Write answer corpus.
    with open(os.path.join(data_path, 'a.corpus'), 'wb') as f:
        pickle.dump(a, f)

    # Write to analyse.
    v.filter_extremes(no_below=1000, keep_n=10000)
    v.compactify()
    v.save(os.path.join(data_path, "raw_vocabulary.gensim"))

def build_embeddings():
    print("Todo: your code goes here")

def build_models():
    print("Todo: your code goes here")    

if __name__ == '__main__':
    
    format_entities()
    build_index()
    build_relations()
    build_vocabulary_and_corpus()

    # build_embeddings()
    # build_models()

