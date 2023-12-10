D_MODEL = 512 # Dimension of input token representation - embeddings
VOCAB_SIZE = 10000 # Number of tokens in vocabulary
MAX_SEQ_LEN = 128 # Maximum Sequence Length of Input and Output Sequence
DROPOUT_P = 0.1 # Quantify the dropout of how much
EPS = 1e-6 # A small value to avoid zero division error while normalization
FF_DROPOUT = 0.2 # How much dropout in FFN