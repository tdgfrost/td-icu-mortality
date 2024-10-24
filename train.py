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
parser.add_argument('--model', type=str,
                    help="Enter one of the following options: TD, 1d, 3d, 7d, 14d, 28d")
parser.add_argument('--hidden_dim', type=int,
                    help="Specify the hidden dimension for the model - default is 32", default=32)
parser.add_argument('--balanced', type=bool,
                    help="Whether to train with balanced cross-entropy", default=False)


def train(train_dataloader, val_dataloader, device, model_type: str, hidden_dim: int, balanced: bool):
    # Define our model class and target label
    if model_type == "TD":
        target_label = "28-day"
        num_models = 2
    else:
        target_label = f"{model_type.rstrip('d')}-day"
        num_models = 1

    # Get the model and target_model
    models = get_model(features=train_dataloader.dataset.features,
                       output_dim=1,
                       feature_scaling=train_dataloader.dataset.scaling,
                       hidden_dim=hidden_dim,
                       dropout=0.5,
                       device=device,
                       num_models=num_models)
    if num_models == 2:
        model, target_model = models
        # Synchronise their parameters
        target_model.load_state_dict(deepcopy(model.state_dict()))
        target_model.train()
    else:
        model = models[0]
        target_model = None

    train_n_segments = train_dataloader.dataset.segments

    # Define the optimizer
    n_tuneable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    learning_rate = 1 / np.sqrt(n_tuneable_params)
    weight_decay = 1 / (learning_rate * train_n_segments)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Define the loss function (with optional class weights)
    all_labels = train_dataloader.dataset.labels[f'{target_label}-died']
    class_weights = 1 / all_labels.unique(return_counts=True)[1]
    class_weights /= class_weights.sum()
    bce_loss = BCELoss(weights_key=class_weights.to(device))

    # Get our evaluation metrics
    metrics = get_metrics()

    # Create a checkpoint directory for the model
    now = datetime.now()
    current_time = now.strftime('%H%M%S')
    current_date = now.strftime('%Y-%m-%d')
    if model_type == "TD":
        model_name = f"{model_type}-{current_time}"
    else:
        if balanced:
            model_name = f"{model_type}-supervised-balanced-{current_time}"
        else:
            model_name = f"{model_type}-supervised-{current_time}"
    checkpoint_dir = f"./models/{current_date}/{model_name}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create Tensorboard for logging
    log_dir = f"./logs/{current_date}/{model_name}"
    writer = SummaryWriter(log_dir=log_dir)
    start_tensorboard()

    # Train the model
    model.train()
    best_val_auroc = -torch.inf
    last_model_name = None

    for epoch in range(10):
        # Perform training loop
        metrics = perform_model_inference_loop(train_dataloader, training_loop=True, model=model,
                                               target_model=target_model, optimizer=optimizer,
                                               loss_fn=bce_loss,
                                               balanced=balanced, target_label=target_label,
                                               metrics=metrics,
                                               epoch=epoch)

        # Perform validation loop
        announce_progress('Validating - all validation data')
        model.eval()
        metrics = perform_model_inference_loop(val_dataloader, training_loop=False,
                                               model=model,
                                               target_model=target_model, optimizer=optimizer,
                                               loss_fn=bce_loss,
                                               balanced=balanced, target_label=target_label,
                                               metrics=metrics,
                                               epoch=epoch)

        # Log the results to tensorboard
        for day in ['1-day', '3-day', '7-day', '14-day', '28-day']:
            writer.add_scalar(f'auroc_{day}', metrics[f'auroc_{day}_results'], epoch)

        writer.flush()

        # Save model checkpoint
        if metrics[f'auroc_{target_label}_results'] > best_val_auroc:
            best_val_auroc = metrics[f'auroc_{target_label}_results']
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                f'auroc_{target_label}': best_val_auroc,
            }, checkpoint_dir + f'/epoch_{epoch + 1}.pt')

            os.remove(last_model_name) if last_model_name else None
            last_model_name = checkpoint_dir + f'/epoch_{epoch + 1}.pt'

        model.train()

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

    # Check appropriate model type chosen
    if args.model not in ['TD', '1d', '3d', '7d', '14d', '28d']:
        raise ValueError('Invalid model type - please choose one of the following: TD, 1d, 3d, 7d, 14d, 28d')

    # Run checks to make sure everything is in place
    check_valid_files_for_training()

    # Get all files required for training
    train_data, val_data = get_training_files(batch_size=64, device=args.device)

    # Train
    train(train_data, val_data, args.device, args.model, args.hidden_dim, args.balanced)
