from galytix_word2vec.config import settings
from galytix_word2vec.phrase_processor import PhraseProcessor
from galytix_word2vec.phraze_vectorizer import PhraseVectorizer


class colors:
    GREEN = "\033[92m"
    ENDC = "\033[0m"
    YELLOW = "\033[93m"


def main():
    processor = PhraseProcessor(
        settings.word2vec_path,
        settings.temp_vectors_path,
        settings.phrases_path,  # noqa: E501
    )
    clean2orig_phrase = processor.clean_phrases()

    vectorizer = PhraseVectorizer(
        processor.word2vec, settings.vector_dim, clean2orig_phrase
    )

    # infinite loop to ask user for input phrase
    print("Enter 'exit' to quit the program.")
    while True:
        phrase = input("Enter a phrase: ")
        if phrase == "exit":
            break
        phrase_vectors_matrix = vectorizer.precompute_phrase_vectors(
            clean2orig_phrase.keys()
        )
        closest_match, score = vectorizer.get_the_closest(
            phrase, phrase_vectors_matrix
        )  # noqa: E501

        print(
            f"The closest match is {colors.GREEN}'{closest_match}'{colors.ENDC} with a similarity score of {colors.YELLOW}{score:.3f}{colors.ENDC}.\n"  # noqa: E501
        )
        print("=" * 50)


if __name__ == "__main__":
    main()
