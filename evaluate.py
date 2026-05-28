import argparse
import os
from datetime import datetime
from utils.train.tools import *
from copy import deepcopy
import torch
import numpy as np
import polars as pl
from torch.utils.tensorboard import SummaryWriter

# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, help="Choose `cpu` or `cuda`", default='cuda')
parser.add_argument('--hidden_dim', type=int,
                    help="Specify the hidden dimension for the model - default is 32", default=32)
parser.add_argument('--model_name', type=str,
                    help="Enter specific model path (e.g., 2026-05-28/TD-162000-seed-42) to evaluate programmatically", default=None)
parser.add_argument('--all', action='store_true',
                    help="Evaluate all trained models in `./models` recursively")


def save_predictions_and_labels(all_predictions, all_labels, dataset_name, model_name):
    """
    Saves collected predictions and true labels to a CSV.
    """
    # Create Polars DataFrame
    data_dict = {
        'prediction': all_predictions
    }
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        col_name = f"label_{day.replace('-day', 'd')}"
        data_dict[col_name] = all_labels[f'{day}-died']
        
    df = pl.DataFrame(data_dict).with_row_index("sample_idx")
    
    # Ensure save directory exists
    os.makedirs('./evaluation_results', exist_ok=True)
    model_name_safe = model_name.replace('/', '_')
    output_path = f'./evaluation_results/{dataset_name}_predictions_{model_name_safe}.csv'
    df.write_csv(output_path)
    print(f"Saved prediction scores and labels to {output_path}")


def evaluate(internal_dataloader, external_dataloader, model_name, device, hidden_dim):
    # Fetch the model
    model = fetch_model(model_name, device, hidden_dim)

    # Get our evaluation metrics
    metrics = get_metrics()

    # Create Tensorboard for logging
    log_dir = f"./logs/{model_name}"
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard(log_dir)

    # Identify if predictions should be flipped (older pre-trained survival models)
    flip_predictions = model_name.startswith("24082")

    # Evaluate the model
    model.eval()
    announce_progress('Validating internally')
    metrics, int_preds, int_labels = perform_model_inference_loop(
        internal_dataloader, training_loop=False,
        model=model, metrics=metrics, flip_predictions=flip_predictions, return_outputs=True
    )

    # Log the internal results to tensorboard
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        writer.add_scalar(f'auroc_{day}_internal_test', metrics[f'auroc_{day}_results'])
    writer.flush()

    # Save internal predictions and labels to CSV
    save_predictions_and_labels(int_preds, int_labels, 'internal', model_name)

    announce_progress('Validating externally')
    metrics, ext_preds, ext_labels = perform_model_inference_loop(
        external_dataloader, training_loop=False,
        model=model, metrics=metrics, flip_predictions=flip_predictions, return_outputs=True
    )

    # Log the exernal results to tensorboard
    for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
        writer.add_scalar(f'auroc_{day}_external_test', metrics[f'auroc_{day}_results'])
    writer.flush()

    # Save external predictions and labels to CSV
    save_predictions_and_labels(ext_preds, ext_labels, 'external', model_name)

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

    # Get all files required for evaluation
    check_valid_files_for_testing()
    internal_data, external_data = get_evaluation_files(batch_size=64, device=args.device)

    # Handle model name selection
    if args.all:
        announce_progress('Scanning for all models in `./models` recursively...')
        model_names = []
        for root, dirs, files in os.walk('./models'):
            if 'checkpoints' in dirs:
                rel_path = os.path.relpath(root, './models')
                checkpoint_path = os.path.join(root, 'checkpoints')
                if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
                    model_names.append(rel_path)
        
        if not model_names:
            print("No models found in `./models` to evaluate recursively.")
        else:
            announce_progress(f"Found {len(model_names)} models to evaluate: {model_names}")
            for model_name in sorted(model_names):
                announce_progress(f"Evaluating model: {model_name}")
                try:
                    evaluate(internal_data, external_data, model_name, args.device, args.hidden_dim)
                except Exception as e:
                    print(f"Error evaluating model {model_name}: {e}")
    elif args.model_name:
        announce_progress(f'Evaluating specified model: {args.model_name}')
        evaluate(internal_data, external_data, args.model_name, args.device, args.hidden_dim)
    else:
        # Get the model path interactively
        announce_progress('Loading model interactively:')
        parent_dir = choose_model_folder('./models')
        model_dir = choose_model_folder(f'./models/{parent_dir}')
        named_model = f'{parent_dir}/{model_dir}'
        evaluate(internal_data, external_data, named_model, args.device, args.hidden_dim)
