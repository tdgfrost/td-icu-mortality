import argparse
import os
import polars as pl
import shutil
import gzip
import time

from utils.parquet_conversion.dtypes import get_dtypes
from utils.preprocessing.tools import announce_progress


# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, help="Choose `mimic` or `sicdb`")
parser.add_argument('--path', type=str, help="Path to `mimic` or `sicdb` data directory")


def convert_mimic_to_parquet(path: str):
    # Check if all target files exist and raise an error if not
    filenames = ['hosp/admissions', 'hosp/patients', 'icu/chartevents', 'icu/d_items', 'icu/inputevents']
    check_file_exists(path, *filenames)

    # Define the target directory
    target_path = './data/mimic/parquet'

    # Begin the conversion process
    convert_all_files(path, target_path, filenames)


def convert_sicdb_to_parquet(path: str):
    # Check if all target files exist and raise an error if not
    filenames = ['cases', 'd_references', 'laboratory', 'medication']
    check_file_exists(path, *filenames)

    # Define the target directory
    target_path = './data/sicdb/parquet'

    # Begin the conversion process
    convert_all_files(path, target_path, filenames)


def check_file_exists(path, *args):
    for arg in args:
        if not os.path.exists(os.path.join(path, arg + '.csv.gz')):
            raise FileNotFoundError(f"File {arg + '.csv.gz'} not found")


def convert_all_files(path, target_path, filenames):
    # Create the target path if it does not exist
    os.makedirs(target_path, exist_ok=True)

    # Get the dtypes for each file
    dtypes_dict = get_dtypes()

    # Begin the conversion process
    start = time.time()
    announce_progress('Beginning conversion...')
    for filename in filenames:
        print(f"Converting {filename}...")
        convert_file(path, target_path, filename, dtypes_dict[filename])
    end = time.time()
    print(f'Conversion completed in {round(end - start, 1)} seconds!')


def convert_file(path, target_path, filename, d_types):
    # Define our paths
    path_csv_gz = os.path.join(path, filename + '.csv.gz')
    path_csv = os.path.join(path, filename + '.csv')

    filename = filename.replace('hosp/', '').replace('icu/', '')
    path_parquet = os.path.join(target_path, filename + '.parquet')

    # (Temporarily) unzip the csv.gz file
    if not os.path.isfile(path_csv):
        with gzip.open(path_csv_gz, 'rb') as f_in, open(path_csv, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    (
        pl.scan_csv(path_csv, schema_overrides=d_types, infer_schema_length=1000000, null_values=[''])
        .sink_parquet(path_parquet)
    )

    # Remove the temporary csv file
    os.remove(path_csv)


if __name__ == "__main__":
    # Parse the command-line arguments inside the block
    args = parser.parse_args()
    if type(args.path) is not str or not os.path.isdir(args.path):
        raise FileNotFoundError("Invalid path argument. Please provide a valid path to the data directory.")

    # Execute the conversion function based on the `type` argument
    if args.type == "mimic":
        convert_mimic_to_parquet(args.path)
    elif args.type == "sicdb":
        convert_sicdb_to_parquet(args.path)
    else:
        raise ValueError("Invalid `type` argument. Choose `mimic` or `sicdb`")
