<<<<<<< HEAD
from config import *
from train_utils import TrainerUtils
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch

class Trainer:

    def __init__(self):

        self.trainer_utils = TrainerUtils()
        self.model, self.train_dataloader, self.test_dataloader, self.writer, self.optimizer = self.trainer_utils.load_train_utils()
        self.epoch_num = 0
        self.step_num = 0
        self.num_epochs = NUM_EPOCHS
        self.model_save_path = MODEL_SAVE_PATH

    def train(self):

        for epoch in range(self.num_epochs):

            for data in tqdm(self.train_dataloader, desc=f"Training epoch {epoch}"):

                loss, self.model, self.optimizer = self.trainer_utils.train_one_step(data, self.model, self.optimizer)

                self.step_num += 1

                self.trainer_utils.log(self.writer, "train_loss", loss, self.step_num)

            self.epoch_num +=1

            saved_message = self.trainer_utils.save_checkpoint(self.model, self.optimizer, self.epoch_num, loss, self.model_save_path)
            print(saved_message)

trainer = Trainer()
trainer.train()



            

=======
import torch
import torch.nn as nn
from config import *
from dataset import ClassificationDataset
from transformer import build_transformer_model
>>>>>>> 8a91c34890dd32bd85267ccb2cb3c2a00c4bd01c
