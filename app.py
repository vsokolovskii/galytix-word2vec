from galytix_word2vec.config import settings
from galytix_word2vec.phrase_processor import PhraseProcessor
from galytix_word2vec.phraze_vectorizer import PhraseVectorizer


def main():
    processor = PhraseProcessor(
        settings.word2vec_path, settings.temp_vectors_path, settings.phrases_path
    )
    cleaned_phrases = processor.clean_phrases()

    vectorizer = PhraseVectorizer(processor.word2vec, settings.vector_dim, cleaned_phrases)

    # infinite loop to ask user for input
    while True:
        phrase = input("Enter a phrase: ")
        if phrase == "exit":
            break
        phrase_vectors_matrix = vectorizer.precompute_phrase_vectors(cleaned_phrases)
        closest_match = vectorizer.get_the_closest(phrase, phrase_vectors_matrix)

        print(f"The closest match to '{phrase}' is '{closest_match}'")

if __name__ == "__main__":
    main()