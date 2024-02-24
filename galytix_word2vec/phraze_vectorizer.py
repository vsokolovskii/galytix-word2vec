import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PhraseVectorizer:
    def __init__(self, word2vec: dict, vector_dim, cleaned2org_phrase: dict):
        """
        Initialize the PhraseVectorizer.

        :param word2vec: A dictionary or a KeyedVectors object containing word vectors.  # noqa: E501
        :param vector_dim: The dimension of the vectors in the word2vec model.
        """
        self.word2vec = word2vec
        self.vector_dim = vector_dim
        self.cleaned_phrases = list(cleaned2org_phrase.keys())
        self.clean2orig_phrase = cleaned2org_phrase

    def phrase_to_vector(self, phrase: str) -> np.ndarray:
        """
        Convert a phrase to a vector by averaging the vectors of the words in the phrase.  # noqa: E501

        :param phrase: A string representing the phrase to be vectorized.
        :return: A numpy array representing the vectorized phrase.
        """
        words = phrase.split()
        word_vectors = np.array(
            [self.word2vec[word] for word in words if word in self.word2vec]
        )

        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.vector_dim)

    def precompute_phrase_vectors(self, phrases: list[str]) -> np.ndarray:
        """
        Precompute the vectors for a list of phrases.

        :param phrases: A list of strings where each string is a phrase to be vectorized.  # noqa: E501
        :return: A numpy array representing the vectors of the phrases.
        """
        phrase_vectors = []
        # Precompute phrase vectors using numpy for efficient computation
        for phrase in phrases:
            phrase_vector = self.phrase_to_vector(phrase)
            phrase_vectors.append(phrase_vector)

        # Convert list of vectors to a numpy matrix
        # for efficient similarity computation
        phrase_vectors_matrix = np.array(phrase_vectors)

        return phrase_vectors_matrix

    def get_the_closest(
        self, phrase: str, phrase_vectors_matrix: np.ndarray
    ) -> [str, float]:
        """
        Find the closest match to a given phrase in the precomputed phrase vectors.  # noqa: E501

        :param phrase: A string representing the phrase to be compared.
        :return: The closest match to the given phrase.
        """
        vector = self.phrase_to_vector(phrase)
        similarity_scores = cosine_similarity([vector], phrase_vectors_matrix)
        closest_match_index = np.argmax(similarity_scores)
        closest_match = self.clean2orig_phrase[
            self.cleaned_phrases[closest_match_index]
        ]

        return closest_match, similarity_scores[0][closest_match_index]
