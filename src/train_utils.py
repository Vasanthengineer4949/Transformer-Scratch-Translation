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
<<<<<<< HEAD
import os
=======
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c

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
<<<<<<< HEAD
        self.vocab_size = VOCAB_SIZE

        self.tokenizer = Tokenizer.from_file(self.tokenizer_path)
        self.transformer_model = build_transformer_model()
        # self.dataset = load_dataset(self.dataset_id, split="train")

        # self.train_dataset, self.test_dataset = self.dataset.train_test_split(test_size=0.15, stratify_by_column=self.tgt_cln_name, shuffle=True)
        self.train_classification_dataset = ClassificationDataset(self.dataset_id, self.tokenizer, self.max_seq_len, self.src_cln_name, self.tgt_cln_name)
        # self.test_classification_dataset = ClassificationDataset(self.test_dataset, self.tokenizer, self.max_seq_len, self.src_cln_name, self.tgt_cln_name)

        self.writer = SummaryWriter(self.logging_dir, comment="First implementation")
        self.optimizer = torch.optim.Adam(self.transformer_model.parameters(), lr = LR)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.token_to_id('[PAD]'), label_smoothing=0.1).to("cpu")

    def load_dataloaders(self):
        train_dataloader = DataLoader(self.train_classification_dataset, batch_size=self.batch_size, shuffle=True)
        # test_dataloader = DataLoader(self.test_classification_dataset, self.batch_size)
        # return train_dataloader, test_dataloader
        return train_dataloader, train_dataloader
=======

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
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c
    
    def compute_loss(self, logits: torch.Tensor, labels: torch.Tensor):
        '''
        Computes Cross Entropy Loss on the model output
        
        Args:
        logits: Model Output Logits(Probabilites)
        labels: Ground Truth
        
        Returns:
        loss: Cross Entropy Loss
        '''
<<<<<<< HEAD
        logits = logits.view(-1, self.vocab_size)
        labels = labels.view(-1)
        loss = self.loss_fn(logits, labels)
        return loss
=======
        logits = logits.view(-1, self.tokenizer.get_vocab_size())
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c

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
    
<<<<<<< HEAD
    def train_one_step(self, model_inp: dict, model: nn.Module, optimizer:torch.optim.Adam):
        
        '''
        Function to do training for one step which will run one forward step compute less and perform backpropagation and step the optimizer to update the parameters
        
        Args:
        model_inp: Batch of dict of model input data which will contain encoder_input_ids, decoder_input_ids, encoder_attention_mask, decoder_attention_mask and labels
        model: Training model
        optimizer: Adam Optimizer

        Returns:
        loss: Loss of that step
        '''

        src_x = model_inp["encoder_input_ids"] 
        src_attn_mask = model_inp["encoder_attention_mask"] 
        tgt_x = model_inp["decoder_input_ids"] 
        tgt_attn_mask = model_inp["decoder_attention_mask"]
        labels = model_inp["labels"]

        model_logits = model(src_x, src_attn_mask, tgt_x, tgt_attn_mask)
        
        loss = self.compute_loss(model_logits, labels)
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        return loss, model, optimizer
    
    def ensure_save_folder(self, path: str):

        '''
        Function to ensure that save folder exists. If folder doesnt exist folder is created
        
        Args:
        path: Model checkpoint saving path
        '''

        if not os.path.exists(path):
            os.makedirs(path) 

    def log(self, writer: SummaryWriter, desc: str, scalar: float, current_step: int):
        
        '''
        Function to log values in the tensorboard
        
        Args:
        writer: Tensorboard Logger - SummaryWriter
        desc: Description of what the value is
        scalar: Value of to be saved
        current_step: Current training step
        '''

        writer.add_scalar(tag=desc, scalar_value=scalar, global_step=current_step)
        writer.flush()

    def save_checkpoint(self, model: nn.Module, optimizer:torch.optim.Adam, epoch_num: int, loss: float, model_save_dir_path: str):

        '''
        Function to save a model checkpoint along with the required relevant information such as the epoch number and loss at the given path
        
        Args:
        model: Transformer model
        optimizer: Adam Optimizer
        epoch_num: Current Epoch finished number
        loss: Loss at the end of the current epoch
        model_save_dir_path: Model saving directory path
        '''

        self.ensure_save_folder(path=model_save_dir_path)
        ckpt_path = model_save_dir_path + f"/epoch-{epoch_num}-checkpoint.pth"
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch_num": epoch_num,
            "loss": loss 
        }, ckpt_path)

        return "Model Checkpoint saved successfully"





        


    
=======
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c
    
    




    