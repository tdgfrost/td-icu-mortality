import os
import polars as pl
import torch
import inquirer
import h5py
from torch.utils.data import IterableDataset, DataLoader
from tensordict import MemoryMappedTensor, TensorDict
from torcheval.metrics import BinaryAUROC
import subprocess
from copy import deepcopy

from utils.train.models import *
from utils.preprocessing.tools import announce_progress, progress_bar


class BCELoss(nn.Module):
    def __init__(self, weights_key=None):
        super(BCELoss, self).__init__()

        self.ce = nn.BCEWithLogitsLoss(reduction='none')
        self.weights_key = weights_key

    def class_weight(self, ce_loss, weights):
        weights = self.weights_key[weights]
        ce_loss *= weights

        return ce_loss

    def forward(self, logits, targets, weights=None):
        """
        Calculate the ordered cross-entropy loss for a set of predictions and targets.
        """
        targets = targets.float().reshape(-1, 1)

        # Calculate BCE loss
        balanced_loss = self.ce(logits, targets)

        # Calculate the class weight
        balanced_loss = self.class_weight(balanced_loss, weights) if weights is not None else balanced_loss

        return balanced_loss.mean()


class Dataset(IterableDataset):
    def __init__(self, data_option, mode, batch_size: int = 64, device: str = 'cpu', verbose: bool = True):
        super(Dataset).__init__()

        self.data, self.labels, self.scaling, self.features, self.label_features = load_torch_binaries(data_option,
                                                                                                       mode)

        self.verbose = verbose
        self.device = device
        self.batch_size = batch_size
        self.n_samples = len(self.labels)
        self.n_features = len(self.features)

        # For the dataloader - identify number of segments
        self.sample_idxs = np.arange(self.n_samples)
        self.segments = self.n_samples // self.batch_size
        self.size = self.segments * self.batch_size

    def __len__(self):
        return self.size

    def __iter__(self):
        return iter(self.generate())

    def generate(self):
        idxs = torch.tensor(np.random.choice(self.sample_idxs, size=(self.segments, self.batch_size),
                                             replace=False),
                            dtype=torch.long, device='cpu')

        for _, preload_step in progress_bar(range(0, len(idxs), 100)) \
                if self.verbose else enumerate(range(0, len(idxs), 100)):
            batch_idxs = idxs[preload_step:preload_step + 100].flatten()
            x, y = self.prepare_data(batch_idxs)

            for batch_step in range(0, len(batch_idxs), self.batch_size):
                yield (x[batch_step:batch_step + self.batch_size],
                       y[batch_step:batch_step + self.batch_size])

    def prepare_data(self, batch_idxs):
        # Code to shift batched data onto the GPU
        x = self.data[batch_idxs].to(self.device)

        # For all our int16 / "-1" missing numbers, switch to NaN
        for i in ['', 'next_']:
            x[i + 'features'] = x[i + 'features'].to(torch.int64)  # <- int64 for indexing requirements much later on
            x[i + 'timepoints'] = torch.where(torch.isnan(x[i + 'values']), torch.nan,
                                              x[i + 'timepoints'].to(torch.float32))
            x[i + 'delta_time'] = torch.where(x[i + 'delta_time'] == -1, torch.nan,
                                              x[i + 'delta_time'].to(torch.float32))

        # Get the labels
        y = self.labels[batch_idxs].to(self.device)

        return x, y


def check_valid_files_for_training():
    for segment in ['train', 'val']:
        if not os.path.isfile(f'./data/mimic/{segment}/h5_array_{segment}.hdf5'):
            raise FileNotFoundError(f'Missing data/mimic/{segment}/h5_array_{segment}.hdf5 '
                                    f'- have you run the convert_to_parquet.py and generate_dataset.py scripts first?')

    return


def check_valid_files_for_testing():
    for data_option in ['mimic', 'sicdb']:
        if not os.path.isfile(f'./data/{data_option}/test/h5_array_test.hdf5'):
            raise FileNotFoundError(f'Missing data/{data_option}/test/h5_array_test.hdf5 '
                                    f'- have you run the convert_to_parquet.py and generate_dataset.py scripts first?')

    return


def choose_model_folder(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    if not folders:
        raise ValueError("No model folders found in `./models` - please ensure you have trained a model first")

    # Use inquirer to let the user select a folder
    question = [inquirer.List('folder',
                              message="Please select folder using UP/DOWN/ENTER",
                              choices=folders)]
    answer = inquirer.prompt(question)
    return answer['folder']


def fetch_model(model_name, device, hidden_dim: int = 32):
    model_path = f'./models/{model_name}/checkpoints'
    files_in_path = os.listdir(model_path)
    if len(files_in_path) == 1:
        model_file = os.path.join(model_path, files_in_path[0])
    else:
        raise ValueError(f'Either no model or more than one file found in {model_path} '
                         f'- please ensure only one file is present')

    with open('./data/features.txt', 'r') as f:
        features = f.read().splitlines()
        f.close()

    scaling = TensorDict.load_memmap(f'./data/scaling_binaries')

    model = get_model(features,
                      output_dim=1,
                      feature_scaling=scaling,
                      hidden_dim=hidden_dim,
                      device=device,
                      dropout=0.5,
                      num_models=1)[0]

    checkpoint = torch.load(model_file, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def get_evaluation_files(batch_size, device: str = 'cpu'):
    # You'll need to get the .hdf5 train/val files, and then retrieve the feature scaling and features from them
    internal_dataset = Dataset('mimic', 'test', batch_size, device)
    external_dataset = Dataset('sicdb', 'test', batch_size, device)

    internal_dataloader = DataLoader(internal_dataset, batch_size=None, collate_fn=lambda x: x)
    external_dataloader = DataLoader(external_dataset, batch_size=None, collate_fn=lambda x: x)
    return internal_dataloader, external_dataloader


def get_metrics():
    return {f'auroc_{i}-day': BinaryAUROC() for i in [1, 3, 7, 14, 28]}


def get_model(features: list, output_dim, feature_scaling, hidden_dim, dropout, device='cpu', num_models: int = 1):
    # Define the model
    models = []
    for _ in range(num_models):
        models.append(CNNLSTMPredictor(n_features=len(features),
                                       features=features,
                                       output_dim=output_dim,
                                       scaling=feature_scaling,
                                       cnn_layers=2,
                                       hidden_dim=hidden_dim,
                                       dropout=dropout,
                                       batch_first=True,
                                       dtype=torch.float32,
                                       device=device))

    return models


def get_training_files(batch_size, device: str = 'cpu'):
    # You'll need to get the .hdf5 train/val files, and then retrieve the feature scaling and features from them
    train_dataset = Dataset('mimic', 'train', batch_size, device, verbose=False)
    val_dataset = Dataset('mimic', 'val', batch_size, device)

    train_dataloader = DataLoader(train_dataset, batch_size=None, collate_fn=lambda x: x)
    val_dataloader = DataLoader(val_dataset, batch_size=None, collate_fn=lambda x: x)
    return train_dataloader, val_dataloader


def has_scaling_binaries():
    return os.path.exists(f'./data/scaling_binaries')


def has_torch_binaries(data_option: str, mode: str):
    binaries_dir = f'./data/{data_option}/{mode}/binaries'

    return all([os.path.exists(os.path.join(binaries_dir, f'{item}'))
                for item in ['data', 'labels']])


def load_torch_binaries(data_option: str, mode: str):
    with open('./data/features.txt', 'r') as f:
        features = f.read().splitlines()
        f.close()

    with open('./data/label_features.txt', 'r') as f:
        label_features = f.read().splitlines()
        f.close()

    if not has_torch_binaries(data_option, mode):
        announce_progress(f'Missing {mode} binary arrays - unpacking them now...')
        unpack_torch_binaries(data_option, mode, features, label_features)

    if not has_scaling_binaries():
        announce_progress(f'Missing scaling binary arrays - unpacking them now...')
        unpack_scaling_binaries(features)

    # Load the torch binaries - keep memory-mapped on CPU for now
    data, labels = [TensorDict.load_memmap(f'./data/{data_option}/{mode}/binaries/{item}')
                    for item in ['data', 'labels']]

    # Load the scaling binaries
    scaling = TensorDict.load_memmap(f'./data/scaling_binaries')

    return data, labels, scaling, features, label_features


def perform_model_inference_loop(dataloader, training_loop: bool = False, model=None, target_model=None, optimizer=None,
                                 loss_fn=None, balanced: bool = False, target_label=None, metrics=None,
                                 epoch: int = None):
    # Make metrics are reset
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        metrics[f'auroc_{day}'].reset()

    balanced_weights = None
    for steps, (inputs, targets) in enumerate(dataloader):

        if training_loop:
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass
            predictions, logits = model(inputs['timepoints'], inputs['values'], inputs['features'],
                                        inputs['delta_time'], inputs['delta_value'])

            if target_model is None:
                target_output = targets[f'{target_label}-died']
            else:
                with torch.no_grad():
                    target_output, _ = target_model(inputs['next_timepoints'], inputs['next_values'],
                                                    inputs['next_features'], inputs['next_delta_time'],
                                                    inputs['next_delta_value'])

                    target_output = torch.where(torch.isnan(inputs['next_values']).all(1),
                                                targets['28-day-died'],
                                                target_output)

            if balanced:
                balanced_weights = targets[f'{target_label}-died'].long()

            # Compute the loss
            loss = loss_fn(logits, target_output, weights=balanced_weights)

            # Backward pass and optimise
            loss.backward()
            optimizer.step()

            # Perform soft target update if required
            if target_model is not None:
                target_model.soft_update(model, alpha=0.99)

        else:
            with torch.no_grad():
                # Forward pass
                predictions, _ = model(inputs['timepoints'], inputs['values'], inputs['features'],
                                       inputs['delta_time'], inputs['delta_value'])

        # Update the running loss and metrics
        for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
            metrics[f'auroc_{day}'].update(predictions.flatten(), targets[f'{day}-died'].long().flatten())

        if training_loop:
            if (steps + 1) % 100 == 0:
                results_str = f"\nEpoch: {epoch + 1}, Batch: {steps + 1}\n"
                for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
                    result = np.round(metrics[f'auroc_{day}'].compute().item(), 5)
                    results_str += f"Avg AUROC ({day}): {result}\n"
                    metrics[f'auroc_{day}'].reset()

                # Print progress
                print(results_str)

    if not training_loop:
        # Calculate the metrics
        results_str = f"Validation evaluation:\n"
        for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
            result = np.round(metrics[f'auroc_{day}'].compute().item(), 5)
            results_str += f"Avg AUROC ({day}): {result}\n"

            metrics[f'auroc_{day}_results'] = result

        # Print progress
        announce_progress(results_str)

    return metrics


def start_tensorboard(logdir='./logs'):
    subprocess.Popen(['tensorboard', '--logdir', logdir, '--load_fast=false'])


def unpack_scaling_binaries(features):
    hdf5_array = h5py.File(f'./data/mimic/train/h5_array_train.hdf5', 'r')

    # Check whether each torch binary already exists - if not, unpack it
    binaries_path = './data/scaling_binaries'

    if not os.path.exists(binaries_path):
        print(f'Unpacking scaling binaries...')
        os.makedirs(binaries_path, exist_ok=True)
        unpack_single_torch_binary(hdf5_array, 'scaling', binaries_path, features, None)

    hdf5_array.close()
    return


def unpack_single_torch_binary(hdf5_array, item: str, binaries_path: str, features: str, label_features: str):
    n_samples = hdf5_array['labels'].shape[0]

    if item == 'scaling':

        data_file = {
            scaling: {embedding: {feature: MemoryMappedTensor.empty(1, dtype=torch.float32,
                                                                    filename=f'{binaries_path}/{scaling}_{embedding}_{feature}.memmap')
                                  for feature in features}
                      for embedding in ['values', 'delta_value', 'delta_time']}
            for scaling in ['min', 'max', 'mean', 'std']
        }
        [data_file[scaling].update({'timepoints': MemoryMappedTensor.empty(1, dtype=torch.float32,
                                                                           filename=f'{binaries_path}/{scaling}_timepoints.memmap')})
         for scaling in ['min', 'max', 'mean', 'std']]

        data_file = TensorDict(data_file, batch_size=(1,), device='cpu').memmap_like(prefix=binaries_path)

        for scaling in ['min', 'max', 'mean', 'std']:
            for embedding in ['values', 'delta_value', 'delta_time', 'timepoints']:
                if embedding == 'timepoints':
                    data_file[scaling][embedding][:] = torch.from_numpy(hdf5_array[scaling][embedding][:])
                    os.remove(f'{binaries_path}/{scaling}_timepoints.memmap')
                else:
                    for feature in features:
                        data_file[scaling][embedding][feature][:] = torch.from_numpy(
                            hdf5_array[scaling][embedding][feature][:])
                        os.remove(f'{binaries_path}/{scaling}_{embedding}_{feature}.memmap')

        return

    elif item == 'labels':

        target_shape = (n_samples, 1)
        data_file = TensorDict({label: MemoryMappedTensor.empty(*target_shape, dtype=torch.int8)
                                for label in label_features},
                               batch_size=target_shape, device='cpu')

        data_file = data_file.memmap_like(prefix=binaries_path)

        for _, chunk in progress_bar(range(0, n_samples, 1000)):
            for idx, label in enumerate(label_features):
                data_file[label][chunk:chunk + 1000] = torch.from_numpy(
                    hdf5_array['labels'][chunk:chunk + 1000, idx]
                ).unsqueeze(-1)
    else:

        target_shape = hdf5_array['values'].shape
        data_file = TensorDict({i + embedding: MemoryMappedTensor.empty(*target_shape, dtype=dtype)
                                for embedding, dtype in [['timepoints', torch.int16],
                                                         ['values', torch.float32],
                                                         ['features', torch.int16],
                                                         ['delta_time', torch.int16],
                                                         ['delta_value', torch.float32]]
                                for i in ['', 'next_']}, device='cpu', batch_size=target_shape)

        data_file = data_file.memmap_like(prefix=binaries_path)

        for _, chunk in progress_bar(range(0, n_samples, 1000)):
            for embedding in ['timepoints', 'values', 'features', 'delta_time', 'delta_value']:
                for i in ['', 'next_']:
                    data_file[i + embedding][chunk:chunk + 1000] = torch.from_numpy(
                        hdf5_array[i + embedding][chunk:chunk + 1000]
                    )

        return


def unpack_torch_binaries(data_option: str, mode: str, features: list, label_features: list):
    hdf5_array = h5py.File(f'./data/{data_option}/{mode}/h5_array_{mode}.hdf5', 'r')

    # Check whether each torch binary already exists - if not, unpack it
    for item in ['data', 'labels']:
        binaries_path = f'./data/{data_option}/{mode}/binaries/{item}'

        if not os.path.exists(binaries_path):
            print(f'Unpacking {item} binary...')
            os.makedirs(binaries_path, exist_ok=True)
            unpack_single_torch_binary(hdf5_array, item, binaries_path, features, label_features)

    hdf5_array.close()
    return
