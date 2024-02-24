import csv
from nltk.corpus import stopwords
from thefuzz import process
from tqdm import tqdm
import pickle
import os
import logging
from gensim.models import KeyedVectors

# set login level as INFO
logging.basicConfig(level=logging.INFO)

class PhraseProcessor:
    def __init__(self, word2vec_path: str, temp_vectors_path: str, phrases_path: str):
        self.word2vec = self.load_word2vec(word2vec_path, temp_vectors_path)
        self.phrases = self.load_phrases(phrases_path)
        self.stopwords = set(stopwords.words("english"))

    @staticmethod
    def load_word2vec(word2vec_path: str, temp_vectors_path: str) -> dict:
        if os.path.exists('word2vec.pkl'):
            logging.info("Loading word2vec from the pickle")
            with open('word2vec.pkl', 'rb') as handle:
                word2vec = pickle.load(handle)
                return word2vec

        if not os.path.exists(temp_vectors_path):
            # Method to load word2vec model, assuming it's stored in a suitable format
            wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=True, limit=1000000
            )
            wv.save_word2vec_format(temp_vectors_path)

        word2vec = {}
        with open(temp_vectors_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header

            for row in tqdm(reader, total=1000000, desc="Loading word2vec to the dictionary"):
                word = row[0]
                try:
                    # Assuming the word vector starts from the second column in the CSV
                    word2vec[word.split()[0]] = list(map(float, word.split()[1:]))
                except (IndexError) as e:
                    continue
        # save dct to the pickle
        with open('word2vec.pkl', 'wb') as handle:
            pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return word2vec

    @staticmethod
    def load_phrases(phrases_path: str) -> list:
        phrases = []
        with open(phrases_path, "r", encoding="ISO-8859-1") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                phrases.append(row["Phrases"])
        return phrases

    def clean_phrases(self) -> list:
        cleaned_phrases = []
        logging.info(f"Cleaning {len(self.phrases)} phrases...")
        for phrase in set(self.phrases):  # Deduplicate
            words = [
                word for word in phrase.split() if word.lower() not in self.stopwords
            ]
            cleaned_words = []
            for word in words:
                if word in self.word2vec:
                    cleaned_words.append(word)
                else:
                    closest_match = process.extractOne(word, self.word2vec.keys())[0]
                    cleaned_words.append(closest_match)
            cleaned_phrases.append(" ".join(cleaned_words))
        logging.info(f"Cleaned {len(cleaned_phrases)} phrases")

        return cleaned_phrases

