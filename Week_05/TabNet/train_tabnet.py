import os
import torch

print('PyTorch Version', torch.__version__)

import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter

from datetime import datetime

from network_tabnet import TabNet
from data_loader import CustomDataset

# Operating with the steps instead of epochs
EPOCHS = 500
EMBEDDING_DIM = 128
BATCH_SIZE = 4096
OPTIMIZER = 'Adam'
LEARNING_RATE = 0.01
WEIGHT_DECAY = 0.0005
LEARNING_RATE_DECAY = 0.4
TRAIN_PATH = '../data/train_adult_cut.pickle'
VALID_PATH = '../data/valid_adult_cut.pickle'
N_STEPS = 5
N_D = 16
N_A = 16
LAMBDA_SPARSE = 0.001
GAMMA = 1.5
LEARNING_RATE_WARMUP_STEPS = 50


class TabNetTrainer(torch.nn.Module):
    def __init__(self):
        super(TabNetTrainer, self).__init__()

        num_workers = 4
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
        self.global_step = 0
        self.best_val_acc = 0

        features_size = len(self.train_dataset.embedding_columns) * EMBEDDING_DIM + \
                        len(self.train_dataset.numeric_columns)
        self.network = TabNet(nrof_unique_categories=self.train_dataset.nrof_unique_categories,
                              embedding_dim=EMBEDDING_DIM,
                              n_d=N_D,
                              n_a=N_A,
                              n_steps=N_STEPS,
                              gamma=GAMMA,
                              features_size=features_size)
        self.network.to('cuda:0')
        # Exponential decay is used

        self.loss = torch.nn.BCEWithLogitsLoss()
        if OPTIMIZER == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER == 'Adagrad':
            self.optimizer = optim.Adagrad(self.network.parameters(), lr=LEARNING_RATE)
        elif OPTIMIZER == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)

        self.lr_exp_decay = ExponentialLR(optimizer=self.optimizer, gamma=0.8)
        self.lr_cos_decay = CosineAnnealingLR(optimizer=self.optimizer, T_max=len(self.train_loader) * EPOCHS)
        return

    def log_params(self):
        text_dict = {
            'EPOCHS': EPOCHS,
            'EMBEDDING_DIM': EMBEDDING_DIM,
            'BATCH_SIZE': BATCH_SIZE,
            'OPTIMIZER': OPTIMIZER,
            'LEARNING_RATE': LEARNING_RATE,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'LEARNING_RATE_DECAY': LEARNING_RATE_DECAY,
            'TRAIN_PATH': TRAIN_PATH,
            'VALID_PATH': VALID_PATH,
            'N_STEPS': N_STEPS,
            'N_D': N_D,
            'N_A': N_A,
            'LAMBDA_SPARSE': LAMBDA_SPARSE,
            'GAMMA': GAMMA,
            'LEARNING_RATE_WARMUP_STEPS': LEARNING_RATE_WARMUP_STEPS,
        }
        l = [key + ": " + str(value) for key, value in text_dict.items()]
        text = '\n'.join(l)
        self.tXw.add_text(tag='Settings', text_string=text)
        return

    def run_train(self):
        print('Run train ...')
        BATCHES = len(self.train_loader)

        for epoch in range(EPOCHS):
            self.network.train()

            tenth = len(self.train_loader) // 10
            for i, (x_num, x_cat, label) in enumerate(self.train_loader):
                if self.global_step <= LEARNING_RATE_WARMUP_STEPS:
                    lr_step = LEARNING_RATE * self.global_step / LEARNING_RATE_WARMUP_STEPS
                    for group in self.optimizer.param_groups:
                        group['lr'] = lr_step
                    print('Setting LR to value', lr_step)
                else:
                    for group in self.optimizer.param_groups:
                        break
                self.tXw.add_scalar('Accuracy/LearningRate', group['lr'], self.global_step)

                # Reset gradients
                self.optimizer.zero_grad()
                # Reshape to the shapes (BatchSize=128, -1)
                BATCH_SIZE = x_num.shape[0]
                x_num = x_num.cuda().view(BATCH_SIZE, -1)
                x_cat = x_cat.cuda().view(BATCH_SIZE, -1)
                # Reshape to the shape (BatchSize, ) because it contains a single label
                label = label.cuda().view(BATCH_SIZE)

                output, masks = self.network(x_num, x_cat)
                # Calculate error and backpropagate
                eps = 0.00001
                binary_loss = self.loss(output, label)
                l_sparse_loss = (-1) * torch.sum(masks * torch.log(masks + eps)) / (N_STEPS * BATCH_SIZE)
                wd_loss = None
                for m in self.modules():
                    m_w_norm = None
                    if isinstance(m, torch.nn.BatchNorm1d):
                        continue
                    elif isinstance(m, torch.nn.Linear):
                        m_w_norm = torch.square(m.weight).sum()
                        if wd_loss is None:
                            wd_loss = m_w_norm
                        else:
                            wd_loss += m_w_norm

                total_loss = binary_loss + LAMBDA_SPARSE * l_sparse_loss + WEIGHT_DECAY * wd_loss
                total_loss.backward()

                output = torch.round(torch.sigmoid(output))
                acc = (output == label).float().mean().item()

                # Update weights with gradients
                self.optimizer.step()
                self.global_step += 1
                if self.global_step % 2500 == 0:
                    self.lr_exp_decay.step()
                # self.lr_cos_decay.step()

                self.tXw.add_scalar('Loss/BinaryLoss', binary_loss.item(), self.global_step)
                self.tXw.add_scalar('Loss/LSparseLoss', l_sparse_loss.item(), self.global_step)
                self.tXw.add_scalar('Loss/WDLoss', wd_loss.item(), self.global_step)
                self.tXw.add_scalar('Loss/TotalLoss', total_loss.item(), self.global_step)
                self.tXw.add_scalar('Accuracy/Train', acc, self.global_step)

                if tenth != 0 and i % tenth == 0:
                    print(f'E [{epoch:3}]/[{EPOCHS:3}] Batch [{i:3}]/[{BATCHES:3}] '
                          f'TotalLoss {total_loss.item():6.5f} Batch Acc {acc:4.3f}')
            # Run validation
            self.network.eval()

            acc_all = torch.zeros(len(self.val_dataset))
            index = 0
            for i, (x_num, x_cat, label) in enumerate(self.val_loader):
                # Reshape to the shapes (BatchSize=128, -1)
                BATCH_SIZE = x_num.shape[0]
                x_num = x_num.cuda().view(BATCH_SIZE, -1)
                x_cat = x_cat.cuda().view(BATCH_SIZE, -1)
                # Reshape to the shape (BatchSize, ) because it contains a single label
                label = label.cuda().view(BATCH_SIZE)

                # Reset gradients
                with torch.no_grad():
                    output, _ = self.network(x_num, x_cat)
                    output = torch.round(torch.sigmoid(output))
                    acc = (output == label).float()
                    acc_all[index:index + BATCH_SIZE] = acc
                    # print(f'Valid. batch {i:3} Accuracy {acc.mean().item():4.3f}')

                index = index + BATCH_SIZE

            acc_val = acc_all.mean().item()
            self.tXw.add_scalar('Accuracy/Val', acc_val, self.global_step)
            self.tXw.flush()
            sfx = f' </> Best {self.best_val_acc:4.3f}'
            if self.best_val_acc + 0.005 < acc_val:
                self.best_val_acc = acc_val
                sfx = ' -> Higher!'

            print(f'E [{epoch:3}]/[{EPOCHS:3}] Val Acc {acc_val:4.3f} {sfx}')

            # if self.global_step > 7700:
            #     print('Finished training by the 7.7K iterations similar to the TabNet paper')
            #     break

        return


if __name__ == '__main__':
    tabnet = TabNetTrainer()
    tabnet.run_train()
