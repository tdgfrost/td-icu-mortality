from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from tensordict import TensorDict

from copy import deepcopy


class MaxPool1D(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPool1D, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, src):
        output = torch.where(torch.isnan(src), torch.tensor(-np.inf, dtype=src.dtype), src)

        output = F.max_pool1d(output, kernel_size=self.kernel_size, stride=self.stride)
        output = torch.where(torch.isinf(output), torch.tensor(np.nan, dtype=output.dtype), output)
        return output


class Transpose(nn.Module):
    """
    A module that transposes the output of a previous layer.
    """

    def __init__(self, dim1, dim2):
        super(Transpose, self).__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


class CNNLSTMPredictor(nn.Module):
    def __init__(self,
                 n_features: int,
                 features: list,
                 output_dim: int,
                 scaling: TensorDict = None,
                 cnn_layers: int = 2,
                 hidden_dim: int = 128,
                 dropout: float = 0.1,
                 batch_first=True,
                 dtype=torch.float32,
                 device='cpu', ):

        super(CNNLSTMPredictor, self).__init__()
        # Save our attributes
        self.dtype = dtype
        self.device = device
        self.n_features = n_features
        self.features = features
        self.output_dim = output_dim

        # Save our scaling values
        self.scaling = scaling

        self.mean_vals = {embedding: torch.concatenate([scaling['mean'][embedding][feature]
                                                        for feature in features]).to(device)
                          for embedding in ['values', 'delta_time', 'delta_value']}
        self.mean_vals.update({'timepoints': scaling['mean']['timepoints'].to(device)})

        self.std_vals = {embedding: torch.concatenate([scaling['std'][embedding][feature]
                                                       for feature in features]).to(device)
                         for embedding in ['values', 'delta_time', 'delta_value']}
        self.std_vals.update({'timepoints': scaling['std']['timepoints'].to(device)})

        # Create our one-to-many embedding networks
        interim_reduced_dim = int(np.sqrt(hidden_dim))
        self.embedding_net = nn.ModuleDict()
        for embedding in ['time', 'value', 'feature', 'delta_time', 'delta_value']:
            if embedding == 'feature':
                self.embedding_net[embedding] = nn.Embedding(n_features, hidden_dim, dtype=dtype
                                                             ).to(device)
                continue
            self.embedding_net[embedding] = nn.Sequential(nn.Linear(1, interim_reduced_dim, dtype=dtype),
                                                          nn.ReLU(),
                                                          nn.Linear(interim_reduced_dim, hidden_dim, dtype=dtype)
                                                          ).to(device)

        self.embedding_norm = nn.Sequential(Transpose(1, 2),
                                            nn.BatchNorm1d(hidden_dim),
                                            Transpose(1, 2)).to(device)

        # Create our CNN layers
        self.cnn_layers = cnn_layers
        self.cnn = nn.Sequential(Transpose(1, 2))
        for _ in range(cnn_layers):
            self.cnn.append(nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=2, stride=1,
                                      padding=1, dtype=dtype))
            self.cnn.append(nn.ReLU())
            self.cnn.append(MaxPool1D(kernel_size=2, stride=2))
        self.cnn.append(Transpose(1, 2))
        self.cnn = self.cnn.to(device)

        # Create our LSTM layers
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim * 8, num_layers=2,
                            dropout=dropout, batch_first=batch_first, dtype=dtype).to(device)

        # Create our dense / decoding layers
        self.dense = nn.Sequential(nn.BatchNorm1d(hidden_dim * 8, dtype=dtype),
                                   nn.Linear(hidden_dim * 8, hidden_dim, dtype=dtype),
                                   nn.ReLU(),
                                   nn.BatchNorm1d(hidden_dim, dtype=dtype),
                                   nn.Linear(hidden_dim, output_dim, dtype=dtype)).to(device)

        # Initialise the weights for all the networks
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def soft_update(self, new_model, alpha=0.99):
        # Update our target network by 1% of the values from the latest prediction network
        update_state_dict = deepcopy(new_model.state_dict())
        with torch.no_grad():
            for key, value in self.state_dict().items():
                update_state_dict[key] = alpha * value + (1 - alpha) * update_state_dict[key]

        self.load_state_dict(update_state_dict)

        return

    def pack_sequences(self, src: torch.Tensor, mask: torch.Tensor = None):
        # Batch, Sequence, Feature
        # Get the lengths
        lengths = (~mask).sum(-1)
        if lengths.min() == 0:  # i.e., when performing inference on the target model on null states
            # Just set the first value as valid (for speed)
            mask[torch.where(lengths == 0)[0], 0] = False
            lengths = torch.where(lengths == 0, 1, lengths)

        # Pack the sequences
        src = pack_padded_sequence(src, lengths.cpu(), batch_first=True, enforce_sorted=False).to(self.device)

        return src

    def get_mask_after_conv(self, mask: torch.Tensor):
        mask = mask.unsqueeze(1).float()
        for _ in range(self.cnn_layers):
            mask = F.avg_pool1d(mask, kernel_size=2, stride=2)
        mask = mask.squeeze(1)

        return mask == 1

    def standardise_inputs(self, timepoints, values, features, delta_time, delta_value):
        # Standardisation of timepoints is straightforward
        standardised_vectors = [(timepoints - self.mean_vals['timepoints']) / self.std_vals['timepoints']]

        # Standardisation of values is more complex and needs to be done on a per-feature basis for each embedding
        features = torch.where(features == -1, 0, features)
        # Create an empty NaN array of (samples, timesteps, n_features)
        long_shape = values.shape[:2] + (self.n_features,)
        torch_nans = torch.ones(long_shape, dtype=self.dtype).to(self.device) * torch.nan
        # Iterate through each embedding
        for embedding, vector in [['values', values], ['delta_time', delta_time], ['delta_value', delta_value]]:
            # Identify all the missing values
            missing_values = torch.isnan(vector)
            # Copy our long NaN array
            new_vector = deepcopy(torch_nans)
            # Fill the NaN array with the observed values (for each feature)
            new_vector.scatter_(-1, features, vector)
            # Standardise on a per-feature basis
            new_vector = (new_vector - self.mean_vals[embedding]) / torch.where(self.std_vals[embedding] == 0, 1,
                                                                          self.std_vals[embedding])
            # Collapse the features back into a single value
            new_vector = torch.nansum(new_vector, -1).unsqueeze(-1)
            # Nansum sets the missing values as "0" - replace these with NaN again
            new_vector = torch.where(missing_values, torch.nan, new_vector)
            # Update our standardised vectors list
            standardised_vectors.extend([new_vector])

        standardised_vectors.extend([features])
        # Returns standardised vectors (new_timepoints, new_values, new_delta_time, new_delta_value, features)
        return tuple(standardised_vectors)

    def forward(self, timepoints: torch.Tensor, values: torch.Tensor, features: torch.Tensor,
                delta_time: torch.Tensor, delta_value: torch.Tensor, normalise: bool = True) -> torch.Tensor:

        # Scale the value and time values
        src = (timepoints, values, features, delta_time, delta_value)

        timepoints, values, delta_time, delta_value, features = self.standardise_inputs(*src) if normalise else src

        # We will re-order the sequence based on timepoints (oldest -> newest)
        # (but keep NaNs at the end, so temporarily assign -inf to these)
        argsort_idx = torch.argsort(torch.where(torch.isnan(timepoints), -torch.inf, timepoints),
                                    dim=1, descending=True)
        timepoints = torch.gather(timepoints.squeeze(), -1, argsort_idx.squeeze())
        values = torch.gather(values.squeeze(), -1, argsort_idx.squeeze())
        features = torch.gather(features.squeeze(), -1, argsort_idx.squeeze())
        delta_time = torch.gather(delta_time.squeeze(), -1, argsort_idx.squeeze())
        delta_value = torch.gather(delta_value.squeeze(), -1, argsort_idx.squeeze())

        # Find the nans and mask them
        src_mask = torch.isnan(timepoints)
        timepoints = torch.where(src_mask, 0, timepoints)
        values = torch.where(src_mask, 0, values)
        features = torch.where(src_mask, 0, features)
        delta_src_mask = torch.isnan(delta_time)
        # These NaNs will get changed to 0 again after the embedding
        delta_time = torch.where(delta_src_mask, 0, delta_time)
        delta_value = torch.where(delta_src_mask, 0, delta_value)

        # Perform embedding
        time_embedding = self.embedding_net['time'](timepoints.unsqueeze(-1))
        value_embedding = self.embedding_net['value'](values.unsqueeze(-1))
        feature_embedding = self.embedding_net['feature'](features)
        delta_time_embedding = self.embedding_net['delta_time'](delta_time.unsqueeze(-1))
        delta_value_embedding = self.embedding_net['delta_value'](delta_value.unsqueeze(-1))

        # For delta - we want NaN's to be 0 AFTER embedding i.e., no effect (but true zero values should be embedded)
        delta_time_embedding = torch.where(delta_src_mask.unsqueeze(-1), 0, delta_time_embedding)
        delta_value_embedding = torch.where(delta_src_mask.unsqueeze(-1), 0, delta_value_embedding)

        embedded_src = time_embedding + value_embedding + feature_embedding + delta_time_embedding + delta_value_embedding
        embedded_src = self.embedding_norm(embedded_src)

        # Perform convolutions
        embedded_src = self.cnn(embedded_src)

        # Update post-cnn NaN mask
        src_mask = self.get_mask_after_conv(src_mask)

        # Run through the LSTM
        if self.device == 'cuda':
            # Packed sequences only optimised for CUDA
            embedded_src = self.pack_sequences(embedded_src, src_mask)
            embedded_src = self.lstm(embedded_src)[1][0][-1]  # Get the last hidden state from the last layer
        else:
            # Technically not "exactly" the same as the above, but ?hopefully close enough in terms of final results
            embedded_src[src_mask] = 0
            embedded_src = self.lstm(embedded_src)[1][0][-1]
            
        # Decode to our prediction and convert to sigmoid
        logits = self.dense(embedded_src)
        predictions = torch.sigmoid(logits.detach())

        return predictions, logits
