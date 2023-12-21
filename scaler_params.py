import re
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from tensorflow.keras.models import load_model
import os
# Read the training dataset
def minmaxscaler_params(path):
    training_set1 = pd.read_csv(path)
    last_column_df = training_set1[['percentRenewable', 'startTime']].copy()
    scaler = MinMaxScaler()
    scaler.fit(last_column_df[['percentRenewable']])
    scaler_dict = {"name":  "MinMaxScaler" , "data_min": list(scaler.data_min_), "data_max": list(scaler.data_max_), "scale": list(scaler.scale_), "min" :list(scaler.min_)}
    return scaler_dict

def standardscaler_params(path):
    training_set1 = pd.read_csv(path)
    last_column_df = training_set1[['percentRenewable', 'startTime']].copy()
    scaler = StandardScaler()
    scaler.fit(last_column_df[['percentRenewable']])
    scaler_dict = {"name":  "StandardScaler" , "scale": list(scaler.scale_), "mean" :list(scaler.mean_)}
    return scaler_dict

def generate_json_from_model_dir():
    training_data_dir = './training_data'
    training_files = os.listdir(training_data_dir)
    training_file_dict = {}
    for t in training_files:
        key = t.split('-')[0]
        training_file_dict[key] = t
    folder_path = "./models"
    models = []
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith(".h5"):
            model_names = filename.split('_')
            country_name = model_names[0]
            scaler = model_names[1]
            seq_length = int(model_names[3].split('.')[0])

            if scaler == 'StandardScaler':
                scaler_params = standardscaler_params(os.path.join('training_data', training_file_dict[country_name]))
            elif scaler == 'MinMaxScaler':
                scaler_params = minmaxscaler_params(os.path.join('training_data', training_file_dict[country_name]))
            else:
                raise ValueError('No such scaler')

            dict = {
                "name": str(filename),
                "version": 1,
                "country": country_name,
                "input_sequence": seq_length,
                "scaler" : scaler_params,
                "description": "Trained on historical data (2020-01-01 - 2023-05-01)"}
            print(dict)
            models.append(dict)
    result = {"models": models}
    return result




# def get_available_country_list():
#     """Returns a list of country codes for which prediction models are available.
#     All models are stored in the 'model' folder. There can be multiple models for one country.
#     This method returns the unique names of all countries for which models exist.
#     """
#     country_names = set()
#     folder_path = "./models"
#     for filename in os.listdir(folder_path):
#         if os.path.isfile(os.path.join(folder_path, filename)) and filename.endswith(".h5"):
#             country_name = filename.split('_')[0]
#             country_names.add(country_name)
#     return list(country_names)

if __name__=="__main__":
    models = generate_json_from_model_dir()
    print(models)
    import json

    with open('models/metadata.json', 'w') as fp:
        json.dump(models, fp)