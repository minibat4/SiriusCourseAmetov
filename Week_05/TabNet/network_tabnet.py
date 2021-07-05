import torch
import torch.nn as nn
from data_loader import CustomDataset

from sparsemax import Sparsemax


class DenseFeatureLayer(nn.Module):
    def __init__(self, nrof_unique_categories, embedding_dim):
        super(DenseFeatureLayer, self).__init__()

        self.embedding_columns = CustomDataset.embedding_columns
        self.numeric_columns = CustomDataset.numeric_columns

        self.embedding_dim = embedding_dim
        self.embeddings = nn.ModuleDict({})
        for i, col in enumerate(self.embedding_columns):
            self.embeddings[col] = torch.nn.Embedding(nrof_unique_categories[col], embedding_dim)

        self.numerical_total_len = len(self.numeric_columns)
        self.embeddings_total_len = len(self.embedding_columns) * self.embedding_dim

        self.feature_bn = torch.nn.BatchNorm1d(self.numerical_total_len + self.embeddings_total_len)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal(m.weight)
                torch.nn.init.zeros_(m.bias)
                print('Initialized a Linear layer module', m)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.xavier_normal_(m.weight)
                print('Initialized an Embedding layer  module', m)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)
                print('Initialized a BatchNorm layer module', m)

        return

    def forward(self, x_num, x_cat):
        x_cat = x_cat.long()

        batch = x_cat.shape[0]
        outputs = torch.FloatTensor(batch, self.embeddings_total_len + self.numerical_total_len)
        outputs = outputs.to(x_cat.device)

        for i, col in enumerate(self.embedding_columns):
            # Assuming x_cat is a batch and it will
            l = i * self.embedding_dim
            r = (i + 1) * self.embedding_dim
            outputs[:, l:r] = self.embeddings[col](x_cat[:, i])

        outputs[:, self.embeddings_total_len:] = x_num
        output = self.feature_bn(outputs)

        return output


class GLULayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(GLULayer, self).__init__()

        self.fc = torch.nn.Linear(input_size, output_size)
        self.fc_bn = torch.nn.BatchNorm1d(output_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, input_data):
        output = self.fc(input_data)
        output = self.fc_bn(output)
        output = torch.nn.functional.glu(output)

        return output


class FeatureTransformer(nn.Module):
    def __init__(self, input_size, output_size, dependent=False):
        super(FeatureTransformer, self).__init__()

        self.scale = torch.sqrt(torch.FloatTensor([0.5]))
        self.dependent = dependent

        self.fc1 = nn.Linear(in_features=input_size, out_features=output_size * 2, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=output_size * 2)
        self.glu1 = GLULayer(input_size=output_size * 2, output_size=output_size * 2)

        self.fc2 = nn.Linear(in_features=output_size, out_features=output_size * 2, bias=True)
        self.bn2 = nn.BatchNorm1d(num_features=output_size * 2)
        self.glu2 = GLULayer(input_size=output_size * 2, output_size=output_size * 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        return

    def forward(self, x):

        output1 = self.fc1(x)
        output1 = self.bn1(output1)
        output1 = self.glu1(output1)

        if self.dependent:
            output1 = (output1 + x) * self.scale.to(x.device)

        output2 = self.fc2(output1)
        output2 = self.bn2(output2)
        output2 = self.glu2(output2)

        # Always
        output2 = (output1 + output2) * self.scale.to(x.device)

        return output2


class AttentiveTransformer(nn.Module):

    def __init__(self, input_size, output_size):
        super(AttentiveTransformer, self).__init__()

        self.fc1 = nn.Linear(in_features=input_size, out_features=output_size, bias=True)
        self.bn1 = nn.BatchNorm1d(num_features=output_size)
        self.sparsemax = Sparsemax(dim=None)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        return

    def forward(self, x, prior_scales):

        x = self.fc1(x)
        x = self.bn1(x)
        x = prior_scales * x
        output = self.sparsemax(x)

        return output


class TabNet(nn.Module):

    def __init__(self, nrof_unique_categories, embedding_dim, n_d, n_a, n_steps, gamma, features_size=133):
        super(TabNet, self).__init__()
        # Define vars
        self.features_size = features_size
        self.nrof_unique_categories = nrof_unique_categories
        self.embedding_dim = embedding_dim
        self.n_d = n_d
        self.n_a = n_a
        n_d_n_a_sum = n_d + n_a
        self.n_d_n_a_sum = n_d_n_a_sum
        self.n_steps = n_steps
        self.gamma = gamma

        # Define layers
        self.dense_feature_layer = DenseFeatureLayer(nrof_unique_categories, embedding_dim)
        self.single_shared_layer = FeatureTransformer(
            input_size=self.features_size,
            output_size=n_d_n_a_sum,
            dependent=False)
        self.first_dependent_layer = FeatureTransformer(
            input_size=n_d_n_a_sum,
            output_size=n_d_n_a_sum,
            dependent=True)
        # Define
        self.multi_dependent_layers = nn.ModuleList()
        for i in range(n_steps):
            m = FeatureTransformer(input_size=n_d_n_a_sum, output_size=n_d_n_a_sum, dependent=True)
            self.multi_dependent_layers.append(m)

        self.attentive_tfs = nn.ModuleList()
        for i in range(n_steps):
            m = AttentiveTransformer(input_size=n_a, output_size=self.features_size)
            self.attentive_tfs.append(m)

        self.layer_out = nn.Linear(in_features=n_d, out_features=1)
        torch.nn.init.xavier_normal_(self.layer_out.weight)
        torch.nn.init.zeros_(self.layer_out.bias)

        return

    def forward(self, x_num, x_cat):
        # Forward through a DenseFeatureTransformer layer
        x_features = self.dense_feature_layer.forward(x_num=x_num, x_cat=x_cat)
        x = self.single_shared_layer(x_features)
        x = self.first_dependent_layer(x)
        # Split the first time
        x_d_out_step = x[:, :self.n_d]
        x_a_out_step = x[:, self.n_d:]
        # Init
        x_d_out = torch.relu(x_d_out_step)
        batch, n_features = x_features.shape
        masks = torch.zeros(self.n_steps, batch, n_features)
        mask = torch.ones_like(x)

        # print('Features before the training loop')
        # print('Features', x_features.shape)
        # print('Shape for N_D part', x_d_out_step.shape)
        # print('Shape for N_A part', x_a_out_step.shape)

        prior_scales = None
        for i in range(self.n_steps):
            # Calculate the prior scales matrix
            if i == 0:
                prior_scales = torch.ones_like(x_features)
            else:
                prior_scales = prior_scales * (self.gamma - mask)
            # Use attentive transformer to calculate the mask
            mask = self.attentive_tfs[i](x_a_out_step, prior_scales=prior_scales)
            masks[i] = mask

            x = x_features * mask
            x = self.single_shared_layer(x)
            x = self.multi_dependent_layers[i](x)
            # Split
            x_d_out_step = x[:, :self.n_d]
            x_a_out_step = x[:, self.n_d:]
            # Compute the common output
            x_d_out = x_d_out + torch.relu(x_d_out_step)

        output = self.layer_out(x_d_out)
        output = output.view(-1)

        return output, masks
