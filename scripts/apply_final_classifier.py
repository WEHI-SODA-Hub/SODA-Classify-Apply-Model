"""
Author: YOKOTE Kenta
Aim: To run the XGBoost classifier on labelled cell data on SLURM

    Takes 8 inputs from STDIN:
        1. run_name: name of the run. The outputs will be saved to a folder
                     which has this as the name
        2. input_folder: 
        3. input_file: name of the file located in input_folder
        4. labels_file: name of the file containing labels 
        5. output_folder: 
        6. classifier_scheme: the type of classifier to use
        7. model_options: 
"""

import sys
import json
import pandas as pd
import pickle
from classifier_initilaliser import ClassifierInitialiser
from preprocess.data_transformer import DataTransformer
import os
from typing import Dict

def apply(run_name: str, 
          input_file: str, 
          input_model: str, 
          output_file: str, 
          preprocess_scheme: str, 
          preprocess_options: Dict, 
          threshold):

    # read the data
    print("INFO: Reading the data")
    X = pd.read_csv(input_file)

    # Read in the model
    print("INFO: Load the model")
    model = pickle.load(open(input_model, 'rb'))

    # Preprocess
    print("INFO: Preprocessing")
    data_transformer = DataTransformer()
    X = data_transformer.transform_data(X, 
                                transform_scheme=preprocess_scheme, 
                                args=preprocess_options)

    # apply the model
    if threshold is None:
        print("INFO: Predicting the labels")
        pd.DataFrame(model.predict(X)).to_csv(output_file)
        print("INFO: Finished")
    else:
        print("INFO: Predicting the labels using threshold")
        probs_df = pd.DataFrame(model.predict_proba(X))
        labels = probs_df.iloc[:, 1] > threshold
        labels = labels.astype(int)
        labels.to_csv(output_file)
        print("INFO: Finished")


if __name__ == '__main__':
    import argparse, toml

    parser = argparse.ArgumentParser(
        prog="MIBI-apply",
        description="This takes an XGBoost classifier model and applies it on unlabelled cell data."
    )

    parser.add_argument("--name", "-n", help="Run name used to label output files.", required=True)
    parser.add_argument("--input", "-i", help="Preprocessed input data file from QuPath.", required=True)
    parser.add_argument("--model", "-m", help="Path to final model file produced from training.", required=True)
    parser.add_argument(
        "--preprocess-scheme",
        "-s",
        help="The scheme to use to transform the input data.",
        choices=["null", "logp1", "poly"], required=True
    )
    parser.add_argument(
        "--options",
        "-x",
        help="Path to TOML file containing preprocessing scheme options.", required=True
    )
    parser.add_argument(
        "--output-path", "-o", help="Path to directory to store output files.", required=True
    )
    parser.add_argument(
        "--threshold", "-t", help="idk what this does yet"
    )

    args = parser.parse_args()

    run_name = args.name
    input_file = args.input
    input_model = args.model
    preprocess_scheme = args.preprocess_scheme
    threshold = args.threshold

    # load options toml
    try:
        preprocess_options = toml.load(args.options)["preprocess_options"]
    except FileNotFoundError:
        print(f"Options TOML file not found at {args.options}")
        sys.exit(2)

    output_file = os.path.join(args.output_path, f"{run_name}_applied_results.csv")

    apply(run_name, input_file, input_model, output_file, preprocess_scheme, preprocess_options, threshold)

