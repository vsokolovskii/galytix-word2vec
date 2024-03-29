import csv
from nltk.corpus import stopwords
from thefuzz import process
from tqdm import tqdm
import pickle
import os
import logging
from gensim.models import KeyedVectors
from galytix_word2vec.config import settings

# set login level as INFO
logging.basicConfig(level=logging.INFO)


class PhraseProcessor:
    def __init__(
        self, word2vec_path: str, temp_vectors_path: str, phrases_path: str
    ):  # noqa: E501
        self.word2vec = self.load_word2vec(word2vec_path, temp_vectors_path)
        self.phrases = self.load_phrases(phrases_path)
        self.stopwords = set(stopwords.words("english"))

    @staticmethod
    def load_word2vec(word2vec_path: str, temp_vectors_path: str) -> dict:
        """
        Loads the word2vec model from the binary file and stores it in a dictionary.  # noqa: E501

        :param word2vec_path: The path to the binary file containing the word2vec mapping  # noqa: E501
        :param temp_vectors_path: The path to the temporary file to store the word2vec csv.  # noqa: E501
        :return: A dictionary containing the word2vec mapping.
        """
        word2vec_pkl_path = os.path.join(settings.data_folder, "word2vec.pkl")
        if (
            os.path.exists(word2vec_pkl_path)
            and settings.skip_word2vec_if_exists is True
        ):
            logging.info("Loading word2vec from the pickle...")
            with open(word2vec_pkl_path, "rb") as handle:
                word2vec = pickle.load(handle)
                return word2vec

        if not os.path.exists(temp_vectors_path):
            wv = KeyedVectors.load_word2vec_format(
                word2vec_path, binary=True, limit=1000000
            )
            wv.save_word2vec_format(temp_vectors_path)

        word2vec = {}
        with open(temp_vectors_path, "r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header

            for row in tqdm(
                reader,
                total=1000000,
                desc="Loading word2vec to the dictionary",  # noqa: E501
            ):
                word = row[0]
                try:
                    word2vec[word.split()[0]] = list(
                        map(float, word.split()[1:])
                    )  # noqa: E501
                except IndexError:
                    continue
        # save dct to the pickle
        with open(word2vec_pkl_path, "wb") as handle:
            pickle.dump(word2vec, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return word2vec

    @staticmethod
    def load_phrases(phrases_path: str) -> list:
        """
        Loads the phrases from the csv file.

        :param phrases_path: The path to the csv file containing the phrases.
        :return: A list of phrases.
        """
        phrases = []
        with open(phrases_path, "r", encoding="ISO-8859-1") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                phrases.append(row["Phrases"])
        return phrases

    def clean_phrases(self) -> dict:
        """
        Cleans the phrases by removing stopwords and words not in the word2vec model.

        :return: A dictionary containing the cleaned phrases as keys and the original phrases as values.  # noqa: E501
        """
        cleaned_phrases_dict = {}
        logging.info(f"Cleaning {len(self.phrases)} phrases...")
        for phrase in set(self.phrases):  # Deduplicate
            original_phrase = phrase  # Store the original phrase
            words = [
                word
                for word in phrase.split()
                if word.lower() not in self.stopwords  # noqa: E501
            ]
            cleaned_words = []
            for word in words:
                if word in self.word2vec:
                    cleaned_words.append(word)
                else:
                    closest_match = process.extractOne(word, self.word2vec.keys())[
                        0
                    ]  # noqa: E501
                    cleaned_words.append(closest_match)
            cleaned_phrase = " ".join(cleaned_words)
            cleaned_phrases_dict[cleaned_phrase] = original_phrase
        logging.info(f"Cleaned {len(cleaned_phrases_dict)} phrases")

        return cleaned_phrases_dict
