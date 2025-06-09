import pickle

def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    data = load_pickle("routerbench_0shot.pkl")
    print(data.info())
    print(data.sample_id.nunique())
    for i in range(1):
        print('sample_id: ', data.iloc[i]['sample_id'])
        # print(data.iloc[i]['prompt'])
        # print(data.iloc[i]['gpt-3.5-turbo-1106'])
        print('eval_name: ', data.iloc[i]['eval_name'])
        print('gpt-3.5-turbo-1106: ', data.iloc[i]['gpt-3.5-turbo-1106'])
        print('gpt-3.5-turbo-1106|model_response: ', data.iloc[i]['gpt-3.5-turbo-1106|model_response'])
        print('gpt-3.5-turbo-1106|total_cost: ', data.iloc[i]['gpt-3.5-turbo-1106|total_cost'])
        print('oracle_model_to_route_to: ', data.iloc[i]['oracle_model_to_route_to'])
    # get sample_id type list
    sample_id_list = data['sample_id'].unique().tolist()
    type_list = set([sample_id.split('.')[0] for sample_id in sample_id_list])
    print(type_list)

    data = load_pickle("routerbench_raw.pkl")
    print(data.info())
    print(data.sample_id.nunique())
    print(data.model_name.unique())
    for i in range(2):
        print('sample_id: ', data.iloc[i]['sample_id'])
        print('eval_name: ', data.iloc[i]['eval_name'])
        print('performance: ', data.iloc[i]['performance'])

