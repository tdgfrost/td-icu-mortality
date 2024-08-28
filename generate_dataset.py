import argparse
import polars as pl

from utils.preprocessing.tools import *


# Define the argument parser and possible arguments globally
parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, help="Choose `mimic` or `sicdb`", default='mimic')
parser.add_argument('--input_window', type=int, default=168,
                    help="Specify the input window size for the model (in hours) - default is 168 hours i.e., 7 days")
parser.add_argument('--context', type=int, default=400,
                    help="Specify the context window size for the model - default is 400")
parser.add_argument('--delay', type=int, default=24,
                    help="Specify the delay (in hours) between states during training - default is 24 hours")
parser.add_argument('--cleanup', type=bool, default=True,
                    help="Deletes the custom parquet files after conversion to .hdf5 - default is True")


def build_mimic_data(input_window_size=168, context_length=400, next_state_delay=24, next_state_window=24):
    # Get our dict of target variables
    target_variables = get_variable_names_for_mimic()

    # Get the required DataFrames
    announce_progress('Loading the data...')
    (admissions, combined_data, patients,
     (train_patient_ids, val_patient_ids, test_patient_ids)) = load_mimic(target_variables)

    # Do the cleaning steps
    announce_progress('Cleaning the data...')
    combined_data = clean_combined_data(combined_data, admissions, train_patient_ids, 'mimic')

    # Get our labels DataFrame
    announce_progress('Creating the labels...')
    labels = create_labels_for_mimic(combined_data, admissions, patients, input_window_size, next_state_delay,
                                     next_state_window)

    # Save the patient IDs
    save_patient_ids_for_mimic(train_patient_ids, val_patient_ids, test_patient_ids)

    # Encode the combined_data DataFrame
    encoded_input_data, features, feature_encoding, encodings = encode_combined_data_for_mimic(combined_data)

    # Get the scaling data
    scaling_data = get_scaling_data_for_mimic(encoded_input_data, labels, train_patient_ids, input_window_size,
                                              encodings['age'], encodings['gender'], encodings['weight'])

    # Save all our data thus far
    combined_data.write_parquet('./data/mimic/parquet/combined_data.parquet')
    encoded_input_data.write_parquet('./data/mimic/parquet/encoded_input_data.parquet')
    feature_encoding.write_parquet('./data/feature_encoding.parquet')
    scaling_data.write_parquet('./data/scaling_data.parquet')
    labels.write_parquet('./data/mimic/parquet/labels.parquet')

    with open('./data/features.txt', 'w') as f:
        for feature in features:
            if feature not in ['subject_id', 'charttime']:
                f.write(feature + '\n') if feature != features[-1] else f.write(feature)
        f.close()

    label_features = ['1-day-died', '3-day-died', '7-day-died', '14-day-died', '28-day-died']
    with open('./data/label_features.txt', 'w') as f:
        for label_feature in label_features:
            f.write(label_feature + '\n') if label_feature != label_features[-1] else f.write(label_feature)
        f.close()

    # Create the finalised input data and label data
    announce_progress('Constructing and saving final dataframes...')
    encoded_input_data = pl.scan_parquet('./data/mimic/parquet/encoded_input_data.parquet').drop('str_feature')
    labels = pl.scan_parquet('./data/mimic/parquet/labels.parquet')
    create_final_dataframe('mimic', encoded_input_data, labels, encodings, train_patient_ids, val_patient_ids, test_patient_ids,
                           ['subject_id', 'labeltime'], ['subject_id', 'labeltime', 'targets'],
                           context_length)

    # Convert these to .hdf5 binaries
    announce_progress('Converting the dataframes to .hdf5 compressed binaries...')
    convert_dataframe_to_hdf5('mimic', context_length)


def build_sicdb_data(input_window_size=168, context_length=400, next_state_delay=24, next_state_window=24):
    # Get our dict of target variables
    target_variables = get_variable_names_for_sicdb()

    # Get the required DataFrames
    announce_progress('Loading the data...')
    cases, combined_data, (train_patient_ids, val_patient_ids, test_patient_ids) = load_sicdb(target_variables)

    # Do the cleaning steps
    announce_progress('Cleaning the data...')
    combined_data = clean_combined_data(combined_data, cases, None, 'sicdb')

    # Get our labels DataFrame
    announce_progress('Creating the labels...')
    labels = create_labels_for_sicdb(combined_data, input_window_size, next_state_delay, next_state_window)

    # Encode the combined_data DataFrame
    encoded_input_data, encodings = encode_combined_data_for_sicdb(combined_data)

    # Save all our data thus far
    combined_data.write_parquet('./data/sicdb/parquet/combined_data.parquet')
    encoded_input_data.write_parquet('./data/sicdb/parquet/encoded_input_data.parquet')
    labels.write_parquet('./data/sicdb/parquet/labels.parquet')

    # Create the finalised input data and label data
    announce_progress('Constructing and saving final dataframes...')
    encoded_input_data = pl.scan_parquet('./data/sicdb/parquet/encoded_input_data.parquet').drop('str_feature')
    labels = pl.scan_parquet('./data/sicdb/parquet/labels.parquet')
    create_final_dataframe('sicdb', encoded_input_data, labels, encodings, train_patient_ids,
                           val_patient_ids, test_patient_ids,
                           ['subject_id', 'CaseID', 'labeltime'],
                           ['subject_id', 'CaseID', 'labeltime', 'targets'],
                           context_length)

    # Convert these to .hdf5 binaries
    announce_progress('Converting the dataframes to .hdf5 compressed binaries...')
    convert_dataframe_to_hdf5('sicdb', context_length)


if __name__ == "__main__":
    # Parse the command-line arguments inside the block
    args = parser.parse_args()

    # Execute the conversion function based on the `type` argument
    if args.type == "mimic":
        build_mimic_data(args.input_window, args.context, args.delay)
        if args.cleanup:
            delete_parquet_files('mimic')
    elif args.type == "sicdb":
        build_sicdb_data(args.input_window, args.context, args.delay)
        if args.cleanup:
            delete_parquet_files('sicdb')
    else:
        raise ValueError("Invalid `type` argument. Choose `mimic` or `sicdb`")
