import string
import spacy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

sp = spacy.load('en_core_web_sm')

def main():
    text_example = "Natural language processing (NLP) is a field " + \
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

    # text = read_file(chapter)
    tokens = tokenize(text_example)
    print(tokens)


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