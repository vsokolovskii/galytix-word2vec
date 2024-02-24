# galytix-word2vec

# Setup
1. `conda create -n galytix python=3.11`
2. `conda activate galytix`
3. `pip install .`

# Running the app
- Place `GoogleNews-vectors-negative300.bin` and `phrases.csv` into the `data` subfolder.
- `python app.py` - running it for the first time will take longer, wav2word needs to be cached
- Enter any phrase, for example, `Trip to Buenos Aires`
