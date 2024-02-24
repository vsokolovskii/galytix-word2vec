import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class PhraseVectorizer:
    def __init__(self, word2vec: dict, vector_dim, phrases: list[str]):
        """
        Initialize the PhraseVectorizer with a word2vec model.
        
        :param word2vec: A dictionary or a KeyedVectors object containing word vectors.
        :param vector_dim: The dimension of the vectors in the word2vec model.
        """
        self.word2vec = word2vec
        self.vector_dim = vector_dim
        self.phrases = phrases

    def phrase_to_vector(self, phrase):
        """
        Convert a phrase to a vector by averaging the vectors of the words in the phrase.

        :param phrase: A string representing the phrase to be vectorized.
        :return: A numpy array representing the vectorized phrase.
        """
        words = phrase.split()
        word_vectors = np.array([self.word2vec[word] for word in words if word in self.word2vec])

        if len(word_vectors) > 0:
            return np.mean(word_vectors, axis=0)
        else:
            return np.zeros(self.vector_dim)
    
    def precompute_phrase_vectors(self, phrases):
        """
        Precompute the vectors for a list of phrases.

        :param phrases: A list of strings where each string is a phrase to be vectorized.
        :return: A numpy array representing the vectors of the phrases.
        """
        phrase_vectors = []
        # Precompute phrase vectors using numpy for efficient computation
        for phrase in phrases:
            phrase_vector = self.phrase_to_vector(phrase)
            phrase_vectors.append(phrase_vector)

        # Convert list of vectors to a numpy matrix for efficient similarity computation
        phrase_vectors_matrix = np.array(phrase_vectors)

        return phrase_vectors_matrix
    
    def get_the_closest(self, phrase, phrase_vectors_matrix):
        """
        Find the closest match to a given phrase in the precomputed phrase vectors.

        :param phrase: A string representing the phrase to be compared.
        :return: The closest match to the given phrase.
        """
        vector = self.phrase_to_vector(phrase)
        similarity_scores = cosine_similarity([vector], phrase_vectors_matrix)
        closest_match_index = np.argmax(similarity_scores)
        closest_match = self.phrases[closest_match_index]

        return closest_match

