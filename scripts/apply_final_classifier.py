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

if __name__ == '__main__':
    # Run options
    run_options_file = sys.argv[1]
    with open(run_options_file) as json_file:
        run_options = json.load(json_file) 

    # Run name
    run_name = run_options["RUN_NAME"]
    
    # Get the input variables 
    input_file = run_options["INPUT_FILE"]

    # The final model
    input_model = run_options["INPUT_MODEL"]

    # Get the output folder
    output_file = run_options["OUTPUT_FILE"]

    # Preprocess
    preprocess_scheme = run_options["PREPROCESS_SCHEME"]
    preprocess_options = run_options["PREPROCESS_OPTIONS"]

    # if a threshold has been defined, use it
    try: 
        threshold = run_options["THRESHOLD"]
    except:
        threshold = None

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



