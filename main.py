"""
Perform Feature Engineering
"""
# Imports
import os
import json
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from utils import is_categorical
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, KFold

# Arguments
parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-3.5-turbo")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--max_sample_nums', type=int, default=20, help='Maximum number of feature engineering samples to generate')
args = parser.parse_args()


if __name__ == '__main__':
    # Define the maximum number of iterations
    global_max_sample_num = args.max_sample_nums
    splits = 5
    seed = 42
    # Load prompt specification
    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()

    problem_name = args.problem_name
    label_encoder = preprocessing.LabelEncoder()
    is_regression = False
    # Add wind power problem to regression list
    if problem_name in ['forest-fires', 'housing', 'insurance', 'bike', 'wine', 'crab'] or 'windpower' in problem_name:
        is_regression = True

    # Load data observations
    file_name = f"./data/{problem_name}.csv"
    df = pd.read_csv(file_name)
    
    target_attr = df.columns[-1]
    is_cat = [is_categorical(df.iloc[:, i]) for i in range(df.shape[1])][:-1]
    attribute_names = df.columns[:-1].tolist()

    X = df.convert_dtypes()
    y = df[target_attr].to_numpy()
    label_list = np.unique(y).tolist()

    X = X.drop(target_attr, axis=1)

    for col in X.columns:
        if X[col].dtype == 'string':
            X[col] = label_encoder.fit_transform(X[col])


    # Handle missing values
    X = X.fillna(0)
    if is_regression == False:
        y = label_encoder.fit_transform(y)
    else:
        y = y
 
    # Load metadata
    meta_data_name = f"./data/{problem_name}-metadata.json"
    meta_data={}
    try:
        with open(meta_data_name, "r") as f:
            filed_meta_data = json.load(f)
    except:
        filed_meta_data = {}
    meta_data = dict(meta_data, **filed_meta_data)

    # Modified for wind power: Use all data instead of outer 5-fold split
    # This allows LLM-FE to train on all available data
    # Original code did nested CV (outer 5-fold + inner 4-fold in spec)
    # which resulted in only 3000 training samples per fold

    # Load config and parameters
    from llmfe import config
    from llmfe import sampler
    from llmfe import evaluator
    from llmfe import pipeline

    class_config = config.ClassConfig(llm_class=sampler.LocalLLM, sandbox_class=evaluator.LocalSandbox)
    config = config.Config(use_api = args.use_api,
                        api_model = args.api_model,)

    # Use ALL data (no outer split)
    # For windpower: extract template (grid1) to pass to LLM-FE
    # Specification will reload full data internally
    if 'windpower' in problem_name:
        from utils.grid_template import extract_grid_template
        X_template = extract_grid_template(X, grid_id=1, source=None)
        data_dict = {'inputs': X_template, 'outputs': y, 'is_cat': is_cat, 'is_regression': is_regression}
    else:
        data_dict = {'inputs': X, 'outputs': y, 'is_cat': is_cat, 'is_regression': is_regression}

    dataset = {'data': data_dict}
    log_path = args.log_path

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        meta_data=meta_data,
        max_sample_nums=global_max_sample_num,  # Changed from global_max_sample_num*splits
        class_config=class_config,
        log_dir=log_path,
    )