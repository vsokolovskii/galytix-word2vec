from setuptools import setup, find_packages

setup(
    name="GalytixWord2Vec",
    version="0.1",
    packages=find_packages(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Vladislav Sokolovskii",
    author_email="sokolovskii.vladislav@gmail.com",
    url="https://github.com/vsokolovskii/galytix-word2vec.git",
    install_requires=[
        "numpy>=1.18.5",  # Assuming you still want numpy as a dependency
        "gensim==4.3.2",
        "pandas==2.2.1",
        "scikit-learn==1.4.0",
        "thefuzz[speedup]==0.22.1",  # Note the [speedup] option for thefuzz
        "nltk==3.8.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
)
