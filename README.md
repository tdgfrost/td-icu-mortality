# td-icu-mortality
Reproducible code for the paper "Robust Mortality Prediction in the Intensive Care Unit using Temporal Difference Learning".

## Introduction

The following is a step-by-step instruction to reproduce the results of my paper. It involves large datasets and a lot of processing - when the data is fully unpacked, at least 55GB will be used and could be closer to 60-70GB. The code assumes a CUDA/NVIDIA GPU, and there are no plans to extend this to Apple Silicon.

The datasets used are [MIMIC-IV v3.0](https://physionet.org/content/mimiciv/3.0/) and [SICdb v1.0.6](https://physionet.org/content/sicdb/1.0.6/) - neither dataset is included in this code and must be acquired separately via PhysioNet's standard pathways. The code has not been tested on later versions of either dataset. If you don't have access to SICDB, you can still perform the training/validation steps on MIMIC.

## Environment

A Python virtual environment should be created and the packages installed as per the following steps (for conda).

1. `conda create --name ENV_NAME --file requirements.txt -c conda-forge`
2. `conda activate ENV_NAME`
3. `pip install torcheval==0.0.7 tensordict==0.5.0`

## Instructions to prepare data prior to pre-processing

For space and performance efficiency, the required files in MIMIC/SICDB will be converted to .parquet format. These converted files will automatically be stored in the local directory, NOT the original MIMIC/SICDB directories.

1. Download (+/- unzip) the MIMIC-IV folder. Do not make any modifications to this folder.

2. `python convert_to_parquet.py --type mimic --path PATH_TO_MIMIC_FOLDER`. In this case, PATH_TO_MIMIC_FOLDER should be the path to the parent directory containing the `hosp` and `icu` folders.

3. (Optional) `python convert_to_parquet.py --type sicdb --path PATH_TO_SICDB_FOLDER` In this case, PATH_TO_SICDB_FOLDER should be the path to the parent directory containing all the `.csv.gz` files.

## Instructions to create the training and testing data

The data is first pre-processed and saved as a Polars DataFrame. It is then automatically split into train/val/test datasets at a default ratio of 80/10/10 (for MIMIC only), and saved as compressed .hdf5 files. 

1. `python generate_dataset.py`
2. (Optional) `python generate_dataset.py`

These are the optional flags:
- The --input_window flag specifies the *eligible* retrospective input timeframe (in hours) - the default is 168 hours (7 days). 
- The --context flag specifies the context length of the input data - the default is 400 input measurements (which includes age, gender, and weight). 
- The --delay flag allows you to set the delay (in hours) between consecutive states, as per the paper - the default is 24 hours.
- The --cleanup flag, if set to True, will conserve space by deleting all customised .parquet files no longer required after successful conversion to .hdf5. If you plan on interrogating any of the customised dataframes directly, you should set this to False - the default setting is True.

## Instructions to train the model

The data will be unpacked as (very large) PyTorch binaries, memory-mapped, and then used for training a specified model, which should be specified using the --model flag ("TD", "1d", "3d", "7d", "14d", or "28d").

1. `python train.py --model TD`

These are the optional flags:
- The --device flag specifies the training device i.e., cpu or cuda - the default is 'cuda'. If 'cuda' is not available, it will default to cpu.
- The --hidden_dim flag specifies the hidden dimensions of the model - the default is 32, as per the original paper.
- The --balanced flag specifies whether to train the model with balanced cross-entropy, as per the paper - this is set to False by default, but should ideally be set to True for the supervised (1d, 3d, etc) models for optimal performance.

## Instructions to evaluate the model

Any pre-trained models can then be evaluated on the MIMIC test and SICDB data. The shell will allow you to select (using the UP/DOWN/ENTER keys) the model you wish to evaluate. The evaluation results will automatically be added to the pre-existing tensorboard logs. N.B. If you have changed the hidden dim of your trained model to a value other than the default of 32, this will need to be specified using the --hidden_dim flag e.g., `--hidden_dim 64`.

1. `python evaluate.py`

If you wish to look at the tensorboard dashboard outside of the train.py and evaluate.py scripts, simply execute the following:

`tensorboard --logdir logs`
