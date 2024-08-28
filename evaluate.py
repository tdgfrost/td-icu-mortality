import argparse
from datetime import datetime
from utils.train.tools import *
from copy import deepcopy
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help="Choose `cpu` or `cuda`", default='cuda')
parser.add_argument('--hidden_dim', type=int,
                    help="Specify the hidden dimension for the model - default is 32", default=32)


def evaluate(internal_dataloader, external_dataloader, model_name, device, hidden_dim):
    # Fetch the model
    model = fetch_model(model_name, device, hidden_dim)

    # Get our evaluation metrics
    metrics = get_metrics()

    # Create Tensorboard for logging
    log_dir = f"./logs/{model_name}"
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)

    # Evaluate the model
    model.eval()
    announce_progress('Validating internally')
    metrics = perform_model_inference_loop(internal_dataloader, training_loop=False,
                                           model=model, metrics=metrics)

    # Log the internal results to tensorboard
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        writer.add_scalar(f'auroc_{day}_internal_test', metrics[f'auroc_{day}_results'])
    writer.flush()

    announce_progress('Validating externally')
    metrics = perform_model_inference_loop(external_dataloader, training_loop=False,
                                           model=model, metrics=metrics)

    # Log the exernal results to tensorboard
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        writer.add_scalar(f'auroc_{day}_external_test', metrics[f'auroc_{day}_results'])
    writer.flush()
    writer.close()


if __name__ == "__main__":
    # Parse the arguments
    args = parser.parse_args()

    # Check device - if cuda not available, set to cpu
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available - switching to CPU')
        args.device = 'cpu'
    elif args.device not in ['cpu', 'cuda']:
        raise ValueError('Invalid device - please choose `cpu` or `cuda`')

    # Get the model path
    announce_progress('Loading model:')
    parent_dir = choose_model_folder('./models')
    model_dir = choose_model_folder(f'./models/{parent_dir}')
    model_path = f'./models/{parent_dir}/{model_dir}/checkpoints'
    named_model = f'{parent_dir}/{model_dir}'
    # named_model = '2024-09-23/TD-103327'

    # Get all files required for evaluation
    check_valid_files_for_testing()
    internal_data, external_data = get_evaluation_files(batch_size=64, device=args.device)

    # Evaluate
    evaluate(internal_data, external_data, named_model, args.device, args.hidden_dim)
