import math
import string

import numpy as np
import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

sp = spacy.load('en_core_web_sm')


def main():
    tokens_example()


def tokens_example():
    text_example_1 = "Natural language processing (NLP) is a field " + \
                     "of computer science, artificial intelligence " + \
                     "and computational linguistics concerned with " + \
                     "the interactions between computers and human " + \
                     "(natural) languages, and, in particular, " + \
                     "concerned with programming computers to " + \
                     "fruitfully process large natural language " + \
                     "corpora. Challenges in natural language " + \
                     "processing frequently involve natural " + \
                     "language understanding, natural language" + \
                     "generation frequently from formal, machine" + \
                     "-readable logical forms), connecting language " + \
                     "and machine perception, managing human-" + \
                     "computer dialog systems, or some combination " + \
                     "thereof."

    text_example_2 = "Natural language processing (NLP) is a subfield of linguistics, computer science, information " \
                     "engineering, and artificial intelligence concerned with the interactions between computers and " \
                     "human (natural) languages, in particular how to program computers to process and analyze large " \
                     "amounts of natural language data." + \
                     "Challenges in natural language processing frequently involve speech recognition, " \
                     "natural language understanding, " \
                     "and natural language generation. "

    hal_example = "the basic concept of the word association"

    # text = read_file(chapter)
    tokens_1 = tokenize(text_example_1)
    tokens_2 = tokenize(text_example_2)

    text_1 = tokens_to_text(tokens_1)
    text_2 = tokens_to_text(tokens_2)

    texts = [text_1, text_2]
    print(tf_idf(texts))

    hal = compute_hal(text_1, 5)
    print(hal)

    top_tokens(text_1)


# noinspection PyUnresolvedReferences
def compute_hal(text, windows_size=2):
    tokens_data = text.split()
    tokens_data_size = len(tokens_data)

    tokens_unique = []
    for t in tokens_data:
        if t not in tokens_unique:
            tokens_unique.append(t)

    tokens_unique_size = len(tokens_unique)

    tokens_map = dict()
    for i in range(tokens_unique_size):
        token = tokens_unique[i]
        tokens_map[token] = i

    hal = pd.DataFrame(data=np.zeros((tokens_unique_size, tokens_unique_size)), index=tokens_unique,
                       columns=tokens_unique)
    for i in range(tokens_data_size):
        target_token = tokens_data[i]
        target_index = tokens_map[target_token]
        for w in range(windows_size + 1):
            if w != 0:
                token_index = i + w
                if 0 <= token_index < tokens_data_size:
                    score = abs(windows_size - abs(w) + 1)

                    window_token = tokens_data[token_index]
                    window_index = tokens_map[window_token]

                    if w > 0:
                        hal.iloc[window_index][target_index] += score
                    # else:
                    #     hal.iloc[target_index][window_index] += score

    return hal


def top_tokens(text, limit=20):
    tokens_data = text.split()

    tokens_map = dict()
    for i in range(len(tokens_data)):
        token = tokens_data[i]
        counter = tokens_map.get(token)
        if counter is None:
            tokens_map[token] = 1
        else:
            tokens_map[token] = counter + 1

    tokens_sorted = {k: v for k, v in sorted(tokens_map.items(), key=lambda item: item[1], reverse=True)[:limit]}
    print(tokens_sorted)
    return tokens_sorted


def tokens_to_text(tokens):
    return ' '.join([' '.join([str(t) for t in elem]) for elem in tokens])


def tokenize(text, language='english'):
    result_tokens = list()
    sentences = sent_tokenize(text)

    for sentence in sentences:
        # lemmatize of words by NLTK
        tokens = lemmatize_spacy(sentence)

        # split into words
        tokens = [w.lower() for w in tokens]

        # remove punctuation from each word
        table = str.maketrans('', '', string.punctuation)
        tokens = [w.translate(table) for w in tokens]

        # remove remaining tokens that are not alphabetic
        tokens = [word for word in tokens if word.isalpha()]

        # filter out stop words
        stop_words = set(stopwords.words(language))
        filtered_words = [w for w in tokens if w not in stop_words]
        result_tokens.append(filtered_words)
    return result_tokens


def tf_idf(docs):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    return pd.DataFrame(dense.tolist(), columns=feature_names)


def lemmatize_spacy(sentence):
    normalized = [word.lemma_ for word in sp(sentence)]
    return normalized


def lemmatize_nltk(sentence, language='english'):
    # split into words
    tokens = word_tokenize(sentence, language)

    # lemmatize of words by NLTK
    lemmatizer = WordNetLemmatizer()
    normalized = [lemmatizer.lemmatize(word) for word in tokens]
    return normalized


def read_file(chapter):
    file = open("book/" + chapter, 'rt')
    text = file.read()
    file.close()
    return text


def get_chapters():
    chapters = list()
    for i in range(8):
        chapters.append("chapter_%d.txt" % i)

    return chapters


if __name__ == '__main__':
    main()
