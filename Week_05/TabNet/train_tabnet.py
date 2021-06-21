import os
import numpy as np
import torch

print('PyTorch Version', torch.__version__)

import torch.optim as optim
import torch.utils.data as data_utils

from tensorboardX import SummaryWriter
from datetime import datetime

from network_tabnet import TabNet
from data_loader import CustomDataset

EPOCHS = 10
EMBEDDING_DIM = 16
BATCH_SIZE = 4096
OPTIMIZER = 'Adam'
LEARNING_RATE = 0.01
TRAIN_PATH = '../data/train_adult_cut.pickle'
VALID_PATH = '../data/valid_adult_cut.pickle'
N_STEPS = 5
N_D = 16
N_A = 16
LAMBDA_SPARSE = 0.001
GAMMA = 1.5


class TabNetTrainer(torch.nn.Module):
    def __init__(self):
        super(TabNetTrainer, self).__init__()

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

        model_path = 'logs/' + datetime.now().__str__().replace(':', '.')[:19]
        if not os.path.exists(model_path):
            os.mkdir(model_path)
            print('Logging to the', model_path)
        self.tXw = SummaryWriter(model_path)
        self.log_params()

        self.network = TabNet(nrof_unique_categories=self.train_dataset.nrof_unique_categories,
                              embedding_dim=EMBEDDING_DIM,
                              n_d=N_D,
                              n_a=N_A,
                              n_steps=N_STEPS,
                              gamma=GAMMA)
        self.network.to('cuda:0')
        # Exponential decay is used

        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.learning_rate_decay = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.8)

        return

    def log_params(self):
        self.tXw.add_hparams(hparam_dict={
            'EPOCHS': EPOCHS,
            'EMBEDDING_DIM': EMBEDDING_DIM,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'TRAIN_PATH': TRAIN_PATH,
            'VALID_PATH': VALID_PATH,
            'N_STEPS': N_STEPS,
            'N_D': N_D,
            'N_A': N_A,
            'LAMBDA_SPARSE': LAMBDA_SPARSE,
            'GAMMA': GAMMA,
        }, metric_dict={})
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
            self.network.train()

            for i, (x_num, x_cat, label) in enumerate(self.train_loader):
                # Reset gradients
                self.optimizer.zero_grad()
                # Reshape to the shapes (BatchSize=128, -1)
                BATCH_SIZE = x_num.shape[0]
                x_num = x_num.to('cuda:0').view(BATCH_SIZE, -1)
                x_cat = x_cat.to('cuda:0').view(BATCH_SIZE, -1)
                # Reshape to the shape (BatchSize, ) because it contains a single label
                label = label.to('cuda:0').view(BATCH_SIZE)

                output = self.network(x_num, x_cat)

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

                print(f'E [{epoch:3}]/[{EPOCHS:3}] Batch [{i:3}]/[{BATCHES:3}] '
                      f'Loss {loss.item():6.5f} Accuracy {acc:4.3f}')

            # self.train_writer.add_histogram('hidden_layer', self.network.linear1.weight.data, self.step)
            print(f'E [{epoch:3}]/[{EPOCHS:3}] Finished!')

            # Run validation
            self.network.eval()
            print('Passing to validation')

            acc_all = 0
            for i, (x_num, x_cat, label) in enumerate(self.val_loader):
                # Reset gradients
                self.optimizer.zero_grad()
                # Reshape to the shapes (BatchSize=128, -1)
                BATCH_SIZE = x_num.shape[0]
                x_num = x_num.to('cuda:0').view(BATCH_SIZE, -1)
                x_cat = x_cat.to('cuda:0').view(BATCH_SIZE, -1)
                # Reshape to the shape (BatchSize, ) because it contains a single label
                label = label.to('cuda:0').view(BATCH_SIZE)

                # Reset gradients
                with torch.no_grad():
                    output = self.network(x_num, x_cat)
                    output = torch.round(torch.sigmoid(output))
                    acc = (output == label).float().mean().item()
                    acc_all += acc * BATCH_SIZE
                    print(f'Valid. batch {i:3} Accuracy {acc:4.3f}')

            acc_val = acc_all / len(self.val_dataset)
            self.tXw.add_scalar('Accuracy/Val', acc_val, self.step)
            print(f'Valid accuracy {acc_val:4.3f}')

        return


if __name__ == '__main__':
    deep_fm = TabNetTrainer()
    deep_fm.run_train()
