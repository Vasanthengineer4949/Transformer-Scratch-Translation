import torch
import torch.nn as nn
from config import *
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from dataset import ClassificationDataset
from transformer import build_transformer_model
from torch.utils.tensorboard import SummaryWriter
from transformer import build_transformer_model

class TrainerUtils:

    def __init__(self):

        '''
        A Utils class to load all the required utilities for model training and holds some helper functions like to compute loss
        '''

        self.dataset_id = DATASET_ID
        self.tokenizer_path = TOKENIZER_PATH
        self.max_seq_len = MAX_SEQ_LEN
        self.src_cln_name = SRC_CLN_NAME
        self.tgt_cln_name = TGT_CLN_NAME
        self.logging_dir = LOG_DIR
        self.batch_size = BATCH_SIZE

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.transformer_model = build_transformer_model()
        self.dataset = load_dataset(self.dataset_id, split="train")["train"]

        self.train_dataset, self.test_dataset = self.dataset.train_test_split(test_size=0.15, stratify_by_column=self.tgt_cln_name, shuffle=True)
        self.train_classification_dataset = ClassificationDataset(self.train_dataset, self.tokenizer, self.max_seq_len, self.src_cln_name, self.tgt_cln_name)
        self.test_classification_dataset = ClassificationDataset(self.test_dataset, self.tokenizer, self.max_seq_len, self.src_cln_name, self.tgt_cln_name)

        self.writer = SummaryWriter(self.logging_dir, comment="First implementation")
        self.optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr = LR)

    def load_dataloaders(self):
        train_dataloader = DataLoader(self.train_classification_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(self.test_classification_dataset, self.batch_size)
        return train_dataloader, test_dataloader
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        '''
        Computes Cross Entropy Loss on the model output
        
        Args:
        logits: Model Output Logits(Probabilites)
        labels: Ground Truth
        
        Returns:
        loss: Cross Entropy Loss
        '''
        logits = logits.view(-1, self.tokenizer.get_vocab_size())

    def generate(self, model: nn.Module, source: torch.Tensor, src_attn_mask: torch.Tensor, tokenizer: Tokenizer):

        '''
        A function to generate output for a given input source data by decoding with the help of greedy decoding.
        
        Args:
        model: Inference model
        source: Source data
        src_attn_mask: Source attention mask
        tokenizer: Tokenizer

        Returns:
        model_out: Model output
        '''

        device = "cpu"

        sos_idx = tokenizer.token_to_id("[SOS]")
        eos_idx = tokenizer.token_to_id("[EOS]")

        decoder_inp = torch.empty(1, 1).type_as(source).fill_(sos_idx).to(device)

        while True:

            if decoder_inp.size(1) == self.max_seq_len:
                break

            tgt_attn_mask = torch.triu(torch.ones((1, decoder_inp.size(1), decoder_inp.size(1))), diagonal=1).type(torch.int)

            out = model(source, src_attn_mask, decoder_inp, tgt_attn_mask)

            logits = torch.softmax(out, dim=1)

            next_word_logit_argmax = torch.argmax(logits, dim=1)

            decoder_inp = torch.cat([decoder_inp, torch.empty(1, 1).type_as(source).fill_(next_word_logit_argmax.item()).to(device)], dim=1)
            
            if next_word_logit_argmax == eos_idx:
                break

        model_out = tokenizer.decode_batch(decoder_inp)

        return model_out

    def load_train_utils(self):
        '''
        A helper function to load all the utilities for training
        
        Returns:
        self.transformer_model: Transformer model
        train_dataloader: Training dataloader
        test_dataloader: Test/Validation dataloader
        self.writer: Tensorboard Logger
        self.optimizer: Model Adam Optimizer
        '''
        train_dataloader, test_dataloader = self.load_dataloaders()
        return self.transformer_model, train_dataloader, test_dataloader, self.writer, self.optimizer
    
    
    




    