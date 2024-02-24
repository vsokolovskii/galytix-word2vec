class Settings:
    app_name: str = "galytix_word2vec"

    # Paths
    word2vec_path: str = (
        "/Users/vladislav/Desktop/galytix-word2vec/GoogleNews-vectors-negative300.bin"  # TODO: Change this to the actual path
    )
    temp_vectors_path = "vectors.csv"
    phrases_path: str = (
        "/Users/vladislav/Desktop/galytix-word2vec/phrases.csv"  # TODO: Change this to the actual path
    )

    # Vectorization
    vector_dim: int = 300

settings = Settings()
