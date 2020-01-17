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
    compute_chapters()


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
    tfidf = tf_idf(texts)
    # print(tfidf)

    hal = compute_hal(text_1, windows_size=5)
    # print(hal)

    top_tokens(text_1)


def compute_chapters():
    chapters = get_chapters()
    texts = list()
    for chapter in chapters:
        raw_text = read_file(chapter)
        print("Raw text for {} file READ".format(chapter))

        tokens = tokenize(raw_text)
        print("Raw text for {} file TOKENIZED".format(chapter))

        text = tokens_to_text(tokens)
        print("Tokens for {} file converted as TEXT".format(chapter))

        hal = compute_hal(text, windows_size=5)
        print("HAL matrix computed for {} text file".format(chapter))

        hal_file = "hal_" + chapter.replace(".txt", ".csv")
        print("Writing HAL matrix to {} file...".format(hal_file))
        hal.to_csv(hal_file, index=None, header=True)
        print("Successfully exported HAL matrix")

        top_tokens_take = 20
        top = list(top_tokens(text, top_tokens_take))

        hal = compute_hal(text, tokens=top, windows_size=5)
        print("HAL matrix computed for top {} tokens for {}".format(top_tokens_take, chapter))

        hal_file = "hal_20" + chapter.replace(".txt", ".csv")
        print("Writing HAL matrix for top {} tokens to {} file...".format(top_tokens_take, hal_file))
        hal.to_csv(hal_file, index=None, header=True)
        print("Successfully exported HAL top {} matrix!\n".format(top_tokens_take))

        texts.append(text)

    tfidf = tf_idf(texts)

    score_file = "tf_idf_chapters.csv"

    print("Writing TF-IDF score to {} file...".format(score_file))
    tfidf.to_csv(score_file, index=None, header=True)
    print("Successfully exported TF-IDF score")


def write_to_file(filename, data):
    file = open(filename, "w+")
    file.write(data)
    file.close()


# noinspection PyUnresolvedReferences
def compute_hal(text, tokens=None, windows_size=2):
    tokens_data = text.split()

    tokens_data_size = len(tokens_data)
    print("Computing HAL matrix for {} tokens".format(tokens_data_size))

    tokens_unique = tokens
    if tokens_unique is None or len(tokens_unique) == 0:
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
        target_index = tokens_map.get(target_token)
        if target_index is not None:
            for w in range(windows_size + 1):
                if w != 0:
                    token_index = i + w
                    if 0 <= token_index < tokens_data_size:
                        score = abs(windows_size - abs(w) + 1)

                        window_token = tokens_data[token_index]
                        window_index = tokens_map.get(window_token)
                        if window_index is not None:
                            if w > 0:
                                hal.iloc[window_index][target_index] += score
                            # else:
                            #     hal.iloc[target_index][window_index] += score

    print("Computed HAL matrix {}x{} for {} tokens".format(tokens_unique_size, tokens_unique_size, tokens_data_size))
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
    print("Picked {} TOP TOKENS out of {} tokens".format(limit, len(tokens_data)))
    print(tokens_sorted)
    return tokens_sorted


def tokens_to_text(tokens):
    print("Converting {} tokens to text".format(sum(map(len, tokens))))
    return ' '.join([' '.join([str(t) for t in elem]) for elem in tokens])


def tokenize(text, language='english'):
    result_tokens = list()
    sentences = sent_tokenize(text)
    print("Tokenizing and lemmatizing {} sentences from given text".format(len(sentences)))

    for sentence in sentences:
        # lemmatize of words
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
        stop_words.add("pron")

        filtered_words = [w for w in tokens if w not in stop_words]
        result_tokens.append(filtered_words)

    print("Finished tokenizing and lemmatizing")
    return result_tokens


def tf_idf(docs):
    print("Computing TF-IDF vectors for {} texts".format(len(docs)))
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(docs)
    feature_names = vectorizer.get_feature_names()

    # sum tfidf frequency of each term through documents
    sums = vectors.sum(axis=0)

    # connecting term to its sums frequency
    data = []
    for col, term in enumerate(feature_names):
        data.append((term, sums[0, col]))

    ranking = pd.DataFrame(data, columns=['term', 'rank'])
    ranking.sort_values('rank', ascending=False)
    score_sorted_file = "tf_idf_sorted_score.csv"

    print("Writing TF-IDF sorted score to {} file...".format(score_sorted_file))
    ranking.to_csv(score_sorted_file, index=None, header=True)

    dense = vectors.todense()
    df = pd.DataFrame(dense.tolist(), columns=feature_names)
    print("Successfully computed TF-IDF vectors")
    return df


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
    print("Starting {} file reading...".format(chapter))
    file = open("book/" + chapter, mode='rt', encoding="utf8")
    text = file.read()
    file.close()
    return text


def get_chapters():
    chapters = list()
    for i in range(1, 9):
        chapters.append("chapter_%d.txt" % i)

    return chapters


if __name__ == '__main__':
    main()
