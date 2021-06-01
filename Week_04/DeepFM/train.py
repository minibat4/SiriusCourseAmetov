import os
import numpy as np
import torch

print(torch.__version__)

import torch.optim as optim
import torch.utils.data as data_utils

# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from pytorch_lightning.metrics import Accuracy
from datetime import datetime

from network import DeepFMNet
from data_loader import CustomDataset

EPOCHS = 10
EMBEDDING_SIZE = 5
BATCH_SIZE = 1024
NROF_LAYERS = 3
NROF_NEURONS = 50
DEEP_OUTPUT_SIZE = 50
NROF_OUT_CLASSES = 1
LEARNING_RATE = 3e-4
TRAIN_PATH = '../data/train_adult.pickle'
VALID_PATH = '../data/valid_adult.pickle'


class DeepFM(torch.nn.Module):
    def __init__(self):
        super(DeepFM, self).__init__()

        num_workers = 0
        pin_memory = True
        self.train_dataset = CustomDataset(TRAIN_PATH)
        self.train_loader = data_utils.DataLoader(dataset=self.train_dataset,
                                                  batch_size=BATCH_SIZE, shuffle=True,
                                                  num_workers=num_workers, pin_memory=pin_memory)
        self.val_dataset = CustomDataset(VALID_PATH)
        self.val_loader = data_utils.DataLoader(dataset=self.val_dataset,
                                                batch_size=BATCH_SIZE, shuffle=False,
                                                num_workers=num_workers, pin_memory=pin_memory)

        self.build_model()
        self.log_params()

        # Prefer to use one writer
        # self.train_writer = SummaryWriter('./logs/train')
        # self.valid_writer = SummaryWriter('./logs/valid')

        model_path = 'logs/' + datetime.now().__str__().replace(':', '.')[:19]
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            print('Logging to the', model_path)
        self.tXw = SummaryWriter(model_path)
        return

    def build_model(self):
        self.network = DeepFMNet(nrof_cat=self.train_dataset.nrof_emb_categories, emb_dim=EMBEDDING_SIZE,
                                 emb_columns=self.train_dataset.embedding_columns,
                                 numeric_columns=self.train_dataset.numeric_columns,
                                 nrof_layers=NROF_LAYERS, nrof_neurons=NROF_NEURONS,
                                 output_size=DEEP_OUTPUT_SIZE,
                                 nrof_out_classes=NROF_OUT_CLASSES)
        self.network.to('cuda:0')

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)

        return

    def log_params(self):
        return

    def load_model(self, restore_path=''):
        if restore_path == '':
            self.step = 0
        else:
            pass

        return

    def run_train(self):
        print('Run train ...')
        BATCHES = len(self.train_loader)

        self.load_model()

        for epoch in range(EPOCHS):
            self.network.turn_train()

            for i, (features, label) in enumerate(self.train_loader):
                # Reset gradients
                for key in features:
                    features[key] = features[key].to('cuda:0')
                label = label.to('cuda:0')

                self.optimizer.zero_grad()

                output = self.network(features)
                # Calculate error and backpropagate
                loss = self.loss(output, label)

                output = torch.round(torch.sigmoid(output))

                loss.backward()
                acc = (output == label).float().mean().item()

                # Update weights with gradients
                self.optimizer.step()

                self.tXw.add_scalar('Loss', loss, self.step)
                self.tXw.add_scalar('Accuracy/Train', acc, self.step)

                self.step += 1

                print(f'E [{epoch:3}]/[{EPOCHS:3}] Batch [{i:3}]/[{BATCHES:3}] Loss {loss.item():6.5f} Accuracy {acc:4.3f}')

            # self.train_writer.add_histogram('hidden_layer', self.network.linear1.weight.data, self.step)
            print(f'E [{epoch:3}]/[{EPOCHS:3}] Finished!')

            # Run validation
            self.network.turn_eval()
            print('Passing to validation')

            acc_all = []
            for i, (features, label) in enumerate(self.val_loader):
                for key in features:
                    features[key] = features[key].to('cuda:0')
                label = label.to('cuda:0')

                # Reset gradients
                with torch.no_grad():
                    output = self.network(features)
                    output = torch.round(torch.sigmoid(output))
                    acc = (output == label).float().mean().item()
                    acc_all.append(acc)
                    print(f'Valid. batch {i:3} Accuracy {acc:4.3f}')

            acc_val = np.mean(acc_all)
            self.tXw.add_scalar('Accuracy/Val', acc_val, self.step)
            print(f'Valid accuracy {acc_val:4.3f}')

        return


deep_fm = DeepFM()
deep_fm.run_train()
