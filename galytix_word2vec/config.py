class Settings:
    app_name: str = "galytix_word2vec"
    # Paths
    data_folder: str = "data"
    word2vec_path: str = f"{data_folder}/GoogleNews-vectors-negative300.bin"
    phrases_path: str = f"{data_folder}/phrases.csv"
    temp_vectors_path = f"{data_folder}/vectors.csv"

    # Vectorization
    vector_dim: int = 300
    skip_word2vec_if_exists: bool = True


settings = Settings()
