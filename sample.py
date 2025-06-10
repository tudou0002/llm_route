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

def scale_cost_columns(df, method='standard'):
    """
    Scale all columns ending with '_cost' together.
    Available methods: 'standard', 'robust', 'minmax', 'maxabs'
    """
    # Get all columns ending with '_cost'
    cost_columns = [col for col in df.columns if col.endswith('_cost')]
    
    if not cost_columns:
        print("No columns ending with '_cost' found")
        return df
    
    # Combine all cost values for global scaling
    all_cost_values = df[cost_columns].values.flatten()
    all_cost_values = all_cost_values[~pd.isna(all_cost_values)]  # Remove NaN values
    
    if method == 'standard':
        # Standard Scaling (Z-score): (x - mean) / std
        global_mean = all_cost_values.mean()
        global_std = all_cost_values.std()
        
        for col in cost_columns:
            df[col] = (df[col] - global_mean) / global_std
        
        print(f"Applied Standard Scaling to {len(cost_columns)} cost columns")
        print(f"Mean: {global_mean:.6f}, Std: {global_std:.6f}")
        
    elif method == 'robust':
        # Robust Scaling: (x - median) / IQR
        global_median = pd.Series(all_cost_values).median()
        q75 = pd.Series(all_cost_values).quantile(0.75)
        q25 = pd.Series(all_cost_values).quantile(0.25)
        iqr = q75 - q25
        
        for col in cost_columns:
            df[col] = (df[col] - global_median) / iqr
        
        print(f"Applied Robust Scaling to {len(cost_columns)} cost columns")
        print(f"Median: {global_median:.6f}, IQR: {iqr:.6f}")
        
    elif method == 'minmax':
        # Min-Max Scaling: (x - min) / (max - min)
        global_min = all_cost_values.min()
        global_max = all_cost_values.max()
        
        for col in cost_columns:
            df[col] = (df[col] - global_min) / (global_max - global_min)
        
        print(f"Applied Min-Max Scaling to {len(cost_columns)} cost columns")
        print(f"Range: [{global_min:.6f}, {global_max:.6f}] -> [0, 1]")
        
    elif method == 'maxabs':
        # Max Absolute Scaling: x / max(|x|)
        global_max_abs = abs(all_cost_values).max()
        
        for col in cost_columns:
            df[col] = df[col] / global_max_abs
        
        print(f"Applied Max Absolute Scaling to {len(cost_columns)} cost columns")
        print(f"Max absolute value: {global_max_abs:.6f}")
        
    else:
        raise ValueError(f"Unknown scaling method: {method}. Use 'standard', 'robust', 'minmax', or 'maxabs'")
    
    return df

def scale_cost_columns_sklearn(df, method='standard'):
    """
    Alternative implementation using sklearn scalers for _cost columns
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, MaxAbsScaler
    
    cost_columns = [col for col in df.columns if col.endswith('_cost')]
    
    if not cost_columns:
        print("No columns ending with '_cost' found")
        return df
    
    # Select the scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'maxabs':
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Fit scaler on all cost data together
    all_cost_data = df[cost_columns].values.reshape(-1, 1)
    all_cost_data = all_cost_data[~pd.isna(all_cost_data)]  # Remove NaN
    
    scaler.fit(all_cost_data)
    
    # Apply scaling to each cost column
    for col in cost_columns:
        df[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
    
    print(f"Applied {method} scaling using sklearn to {len(cost_columns)} cost columns")
    
    return df

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
    
    # Scale all cost columns together
    result_df = scale_cost_columns(result_df, method='standard')  # You can change the method here
    
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