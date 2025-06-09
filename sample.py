import pandas as pd
import random
import pickle
import argparse

MODELS = ['gpt-3.5-turbo-1106', 'claude-instant-v1', 'claude-v1', 'claude-v2',  
 'gpt-4-1106-preview', 'meta/llama-2-70b-chat',
 'mistralai/mixtral-8x7b-chat', 'zero-one-ai/Yi-34B-Chat',
 'WizardLM/WizardLM-13B-V1.2', 'meta/code-llama-instruct-34b-chat',
 'mistralai/mistral-7b-chat']

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def filter_data(args):
    data = load_pickle(args.filename)
    if args.split == "dev":
        return data[data['sample_id'].str.contains('.dev.')]
    elif args.split == "test":
        return data[~data['sample_id'].str.contains('.test.')]
    elif args.split == "val":
        return data[data['sample_id'].str.contains('.val.')]
    else:
        raise ValueError(f"Invalid split: {args.split}")

def sample_data(data):
    # set seed for reproduction
    random.seed(42)
    sample_id_list = data['sample_id'].unique().tolist()
    sample_id_list = random.sample(sample_id_list, args.num_samples)

    # preprocess prompt column to be str
    for col in ['prompt', 'model_response']:
        data[col] = data[col].astype(str)
        data[col] = data[col].str.strip('[')
        data[col] = data[col].str.strip(']')
        data[col] = data[col].str.strip('"')
        data[col] = data[col].str.strip("'")
    return data[data['sample_id'].isin(sample_id_list)]

def reformat_data(df):
    # Pivot multiple columns based on the model column
    # This will create columns like 'gpt-3.5-turbo-1106_performance', 'gpt-3.5-turbo-1106_response', etc.
    
    # Create pivot tables for each column we want to pivot
    performance_pivot = df.pivot_table(
        index='sample_id', 
        columns='model_name', 
        values='performance', 
        aggfunc='first'
    )
    
    response_pivot = df.pivot_table(
        index='sample_id', 
        columns='model_name', 
        values='model_response', 
        aggfunc='first'
    )
    
    cost_pivot = df.pivot_table(
        index='sample_id', 
        columns='model_name', 
        values='cost', 
        aggfunc='first'
    )
    
    # Rename columns to add appropriate suffixes
    performance_pivot.columns = [f"{col}_performance" for col in performance_pivot.columns]
    response_pivot.columns = [f"{col}_response" for col in response_pivot.columns]
    cost_pivot.columns = [f"{col}_cost" for col in cost_pivot.columns]
    
    # Get other columns (non-pivoted data) by taking first value for each sample_id
    other_data = df.groupby('sample_id').agg({
        'eval_name': 'first',
        'prompt': 'first',
        # Add any other columns that should be aggregated with 'first'
    }).reset_index()
    
    # Merge all pivot tables with other data
    result_df = other_data.merge(performance_pivot, on='sample_id', how='left')
    result_df = result_df.merge(response_pivot, on='sample_id', how='left')
    result_df = result_df.merge(cost_pivot, on='sample_id', how='left')
    
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="routerbench_raw.pkl")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--split", type=str, default="dev")
    parser.add_argument("--o", type=str, default="routerbench_sample.csv")
    args = parser.parse_args()

    data = filter_data(args)
    sample_df = sample_data(data)
    sample_df = reformat_data(sample_df)
    print(sample_df.info())
    sample_df.to_csv(args.o, index=False)