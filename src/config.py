D_MODEL = 512 # Dimension of input token representation - embeddings
VOCAB_SIZE = 50000 # Number of tokens in vocabulary
MAX_SEQ_LEN = 128 # Maximum Sequence Length of Input and Output Sequence
NUM_LAYERS = 6 # Number of encoder and decoder stacks
ATTN_DROPOUT = 0.2 # Quantify the dropout of how much in attention
EPS = 1e-6 # A small value to avoid zero division error while normalization
FF_DROPOUT = 0.2 # How much dropout in FFN``
NUM_HEADS = 8 # Number of attention heads
RES_DROPOUT = 0.2 # Dropout for residual connection
<<<<<<< HEAD
TOKENIZER_PATH = "tokenizer.json" # Tokenizer path
DATASET_ID = "data.parquet" # Dataset path
=======
TOKENIZER_PATH = "Transformer-Scratch-Translation/src/tokenizer.json" # Tokenizer path
DATASET_ID = "imdb" # Dataset id
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c
SRC_CLN_NAME = "text" # Source column name
TGT_CLN_NAME = "label" # Target column name
LOG_DIR = "logs/v1" # Logging dir
BATCH_SIZE = 10 # Number of samples per batch
<<<<<<< HEAD
LR = 1e-4 # Learning Rate for optimizer
NUM_EPOCHS = 5 # Number of training epochs
MODEL_SAVE_PATH = "transformer_classifier" # Model save path
=======
LR = 1e-4 # Learning Rate for optimizer
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c
