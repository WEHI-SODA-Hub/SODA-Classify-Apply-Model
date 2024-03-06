"""
Author: YOKOTE Kenta
"""

import sys
import pandas as pd
import pickle
from preprocess.data_transformer import DataTransformer
import os
from typing import Dict
import json
import tabulate


def apply(
    input_file: str,
    input_model: str,
    output_file: str,
    preprocess_scheme: str,
    preprocess_options: Dict,
    decoder_file: str,
    images_file: str,
    validation_file: str,
    threshold,
) -> None:
    """
    main function to apply an XGBoost classifier model to unlabelled cell type data

    Args:
        input_file: Preprocessed input data file path from QuPath.
        input_model: Path to final model file produced from training.
        output_file: Path to applied model results.
        preprocess_scheme: The scheme to use to transform the input data.
        preprocess_options: Dict containing preprocessing scheme options.
        decoder_file: Path to JSON file decoder
        images_file: Path to images and coordinate columns CSV file.
        validation_file: Path to preprocessed input data used to train the XGBoost model.
        threshold: not sure yet

    Raises:
        FileNotFoundError if any of input_file, input_model are not found.
    """

    # read the data
    print("INFO: Reading the data")
    X = pd.read_csv(input_file)

    print("INFO: Validating input file")
    # get the columns of the validation csv
    cols_val = pd.read_csv(validation_file, nrows=0).columns
    cols_in = X.columns
    # check whether they share the same columns
    # assumes there is no duplication, which I think is a safe assumption if the data is processed through MIBI-preprocess-data
    if len(cols_in) == len(cols_in.intersection(cols_val)):
        print("INFO: Input and validation files' columns are in a different order.\n      Input data will be reordered before applying the model.")
        X = X[cols_val] # ensures that input data's columns are in the right order 
    else:
        # find which columns don't match
        cols_in_input = cols_in[~cols_in.isin(cols_val)]
        cols_in_val = cols_val[~cols_val.isin(cols_in)]
        print("It looks like the columns in the input file don't match the validation file!\n",
            tabulate.tabulate({
                "Columns in Input File, but not in Validation File": cols_in_input, 
                "Columns in Validation File, but not in Input File": cols_in_val
            }, headers="keys"),
            sep='\n'
        )
        sys.exit(1)

    # Read in the model
    print("INFO: Load the model")
    model = pickle.load(open(input_model, "rb"))

    # Preprocess
    print("INFO: Preprocessing")
    data_transformer = DataTransformer()
    X = data_transformer.transform_data(
        X, transform_scheme=preprocess_scheme, args=preprocess_options
    )

    # apply the model
    if threshold:
        print("INFO: Predicting the labels using threshold")
        probs_df = pd.DataFrame(model.predict_proba(X))
        y_pred = probs_df.iloc[:, 1] > threshold
        y_pred = y_pred.astype(int)
        y_pred.to_csv(output_file)
        print("INFO: Finished")
    else:
        print("INFO: Predicting the labels")
        y_pred = pd.DataFrame(model.predict(X))

        print("INFO: Load the decoder")
        try:
            decoder = json.load(open(decoder_file, "r"))
        except FileNotFoundError:
            raise FileNotFoundError(f"Could not find decoder file at {decoder_file}.")

        print("INFO: Load the images and coordinate columns CSV file")
        try:
            images_coordinates = pd.read_csv(images_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find images and coordinate columns CSV at {images_file}."
            )

        print("INFO: Converting predicted to QuPath-compatible format")
        final_labels = images_coordinates.copy()
        final_labels.loc[:, "Prediction Level 1"] = (
            y_pred.iloc[:, 0].astype("str").replace(decoder)
        )
        final_labels.to_csv(output_file)

        print("INFO: Finished")


if __name__ == "__main__":
    import argparse, toml

    parser = argparse.ArgumentParser(
        prog="MIBI-apply",
        description="This takes an XGBoost classifier model and applies it on unlabelled cell data.",
    )

    parser.add_argument(
        "--name", "-n", help="Run name used to label output files.", required=True
    )
    parser.add_argument(
        "--input", "-i", help="Preprocessed input data file from QuPath.", required=True
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Path to final model file produced from training.",
        required=True,
    )
    parser.add_argument(
        "--preprocess-scheme",
        "-s",
        help="The scheme to use to transform the input data.",
        choices=["null", "logp1", "poly"],
        required=True,
    )
    parser.add_argument(
        "--options",
        "-x",
        help="Path to TOML file containing preprocessing scheme options.",
    )
    parser.add_argument(
        "--decoder",
        "-d",
        help="Path to decoder JSON file. Used to match predicted values to their cell names",
        required=True,
    )
    parser.add_argument(
        "--images-file",
        "-f",
        help="Path to images and coordinate columns CSV file. Used when converting predicted results back to QuPath-compatible format.",
        required=True,
    )
    parser.add_argument(
        "--validation-file",
        "-v",
        help="Path to preprocessed input data used to train the XGBoost model. This is used to validate the input data.",
        required=True
    )
    parser.add_argument(
        "--output-path",
        "-o",
        help="Path to directory to store output files.",
        required=True,
    )
    parser.add_argument("--threshold", "-t", help="idk what this does yet", type=float)

    args, unknown_args = parser.parse_known_args()

    if args.preprocess_scheme == "poly":
        if not args.options:
            parser.error("--options is required when --preprocess-scheme is poly.")

    args = parser.parse_args()

    run_name = args.name
    input_file = args.input
    input_model = args.model
    preprocess_scheme = args.preprocess_scheme
    decoder = args.decoder
    images_file = args.images_file
    validation_file = args.validation_file
    threshold = args.threshold

    # load options toml
    preprocess_options = args.options
    if preprocess_options:
        try:
            preprocess_options = toml.load(preprocess_options)["preprocess_options"]
        except FileNotFoundError:
            print(f"Options TOML file not found at {preprocess_options}")
            sys.exit(2)

    output_file = os.path.join(args.output_path, f"{run_name}_applied_results.csv")

    apply(
        input_file,
        input_model,
        output_file,
        preprocess_scheme,
        preprocess_options,
        decoder,
        images_file,
        validation_file,
        threshold,
    )
