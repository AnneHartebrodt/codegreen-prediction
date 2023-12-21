"""
This file contains methods for interacting with the models stored in the "model" folder.
These methods include retrieving forecast data from the ENTSOE portal (which serves as 
input for prediction models) , running models, finding the latest models for a specific 
country, and obtaining a list of countries for which models are available.

The main method is `model_run_latest(country)`.
"""

import os
import json
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import re

import entsoeAPI as en
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def get_model_metadata(model):
    """Returns metadata for the selected model from the metadata.json file in the model folder"""
    with open("./models/metadata.json", "r") as file:

        data = json.load(file)
        obj = [o for o in data["models"] if o["name"] == model]
        if len(obj) == 1:
            return obj[0]
        else:
            raise Exception("Invalid model name")


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

def get_available_country_list():
    """Returns a list of country codes for which prediction models are available.
    All models are stored in the 'model' folder. There can be multiple models for one country.
    This method returns the unique names of all countries for which models exist.
    """
    country_names = set()
    print('Getting countries')
    with open("./models/metadata.json", "r") as file:
        data = json.load(file)
        obj = [o["country"] for o in data['models']]
    return obj

def get_latest_model_name_for(country):
    """Returns the latest prediction model version number for a country.
    All models stored in the 'model' folder follow a common file naming convention: "countrycode_version".
    This method returns the value of the highest version available for the given country.
    """
    highestNumber = float('-inf')  # Start with a very low value
    highestNumberFile = None
    for fileName in os.listdir("./models"):
        if fileName.startswith(country + "_v") and fileName.endswith(".h5"):
            fileNumber = int(fileName.split("_")[1].split(".")[0][1:])
            if fileNumber > highestNumber:
                highestNumber = fileNumber
                highestNumberFile = fileName
    return highestNumberFile

def get_latest_model_name_for(country):
    """Returns the latest prediction model version number for a country.
    All models stored in the 'model' folder follow a common file naming convention: "countrycode_version".
    This method returns the value of the highest version available for the given country.
    """
    last_version  = -1
    filename = None
    with open("./models/metadata.json", "r") as file:
        data = json.load(file)['models']
        for d in data:
            if d['country'] == country:
                if d['version']>last_version:
                    last_version = d['version']
                    filename = d['name']
    return filename, last_version



def get_date_range(input_sequence):
    """Returns a dictionary comprising two keys: 'start' and 'end'. 
    These values are used as the start and end dates to retrieve actual generation data from the ENTSOE API.
    The 'start' date is established as 3 days before the current date, ensuring a comprehensive historical range. 
    The 'end' date aligns with the start of the last hour, which guarantees data retrieval up to 2 hours before the present hour. 
    According to the ENTSOE regulation, actual generation values shall be published on later than one after the operational period
    As various countries report data in either 15-minute or 60-minute intervals, it's prudent to assume that accurate data is available 
    for 2 hours prior to the current hour. 
    Both dates adhere to the format YYYYMMDDhhmm, with the 'hhmm' portion set as '0000'.    
    For instance, if the current time is 14:34, the end date will be 13:00 of the current day, encompassing data up to the preceding hour.
    """
    today_utc = datetime.now()
    days = input_sequence/24
    start_date = (today_utc - timedelta(days=days)).replace(hour=0,
                                                         minute=0, second=0, microsecond=0)
    end_date = (today_utc - timedelta(hours=1)
                ).replace(minute=0, second=0, microsecond=0)
    start_date_str = start_date.strftime('%Y%m%d%H%M')
    end_date_str = end_date.strftime('%Y%m%d%H%M')
    date_range = {"start": start_date_str, "end": end_date_str}
    return date_range


def get_percent_actual_generation(country, input_sequence):
    ''' Returns a pandas DataFrame of the hourly actual percentage of renewable energy collected from the ENTSOE portal for a 
    specified country over the last n hours. The last hour will be the current hour or hour upto which data is available. 
    The value of n is determined by the input_sequence provided.
    The output from this method serves as input for running the model.
    '''
    input = get_date_range(input_sequence)
    data = en.get_actual_percent_renewable(
        country, input["start"], input["end"], True)
    # data.to_csv("./data/test-"+country+".csv")
    last_n_rows = data.tail(input_sequence)
    return last_n_rows


def predict(model_name, last_values, scaler, seq_len):
    """
    Predicts the next 48 hours of percent renewable energy based on a pre-trained model.

    Args:
        model_name (str): The name of the pre-trained model file.
        last_values (pd.DataFrame): DataFrame containing the last values of percentRenewable and startTime.

    Returns:
        pd.DataFrame: DataFrame containing the forecast values and timestamps.
    """
    # Extract scaling technique and sequence length from the model name
    last_values_subset = last_values[['percentRenewable', 'startTimeUTC']].copy()
    last_values_subset['startTimeUTC'] = pd.to_datetime(last_values_subset['startTimeUTC'], format='%Y%m%d%H%M')

    # Extract the last timestamp from the input data
    last_timestamp = last_values_subset['startTimeUTC'].iloc[-1]

    # Extract sequence length from the model name

    model_filename = "./models/"+model_name
    # Load the specified model
    #lstm = load_model(model_filename, compile=False)
    # Load the pre-trained model
    model = load_model(model_filename, compile=False)

    # Extract the last (seq_len-1) values from last_values
    last_values = last_values['percentRenewable'].tail(seq_len - 1).values.flatten()

    # Initialize the scaler based on the scaling techniq
    # List to store the forecast values
    forecast_values = []

    # Generate forecasts for the next 48 hours
    for _ in range(48):
        # Scale the last values
        scaled_last_values = scaler.transform(last_values.reshape(-1, 1))

        # Prepare the input for prediction
        x_pred = scaled_last_values[-(seq_len - 1):].reshape(1, (seq_len - 1), 1)

        # Predict the next value
        predicted_value = model.predict(x_pred)

        # Inverse transform the predicted value
        predicted_value = scaler.inverse_transform(predicted_value)

        # Append the predicted value to the forecast_values
        forecast_values.append(predicted_value[0][0])

        # Update last_values with the predicted value
        last_values = np.append(last_values, predicted_value)

    # Generate the next 48 timestamps
    forecast_timestamps = pd.date_range(start=last_timestamp, periods=49, freq='H')[1:]

    # Create a DataFrame with forecast values and timestamps
    forecast_df = pd.DataFrame({'startTimeUTC': forecast_timestamps, 'percentRenewableForecast': forecast_values})
    return forecast_df

def get_scaler(model_meta):
    """
    Initialized the scaler from the metadata
    """
    if model_meta['scaler']['name'] == 'StandardScaler':
        # reinitialize scaler
        new_scaler = StandardScaler(with_mean=False, with_std=False)
        new_scaler.mean_ = model_meta['scaler']['mean']
        new_scaler.scale_ = model_meta['scaler']['scale']

    elif model_meta['scaler']['name'] == 'MinMaxScaler':
        new_scaler = MinMaxScaler(feature_range=(0, 1))
        new_scaler.data_min_ = model_meta['scaler']['data_min']
        new_scaler.data_max_ = model_meta['scaler']['data_max']
        new_scaler.scale_ = model_meta['scaler']['scale']
        new_scaler.min_ = model_meta['scaler']['min']

    else:
        raise ValueError('Invalid Scaler name')

    return new_scaler

def run_latest_model(country) -> dict:
    """ Returns  predictions by running the latest version of model available for the input country
    :param country : 2 letter country code
    :type country : str
    :return Dictionary { "input": { "country":"", "model":"", "start":"", "end":"",  "percentRenewable":[],  } , "output": <pandas dataframe> }
    """
    # get the name of the latest model  and its metadata
    model_name, version = get_latest_model_name_for(country)
    model_meta = get_model_metadata(model_name)

    input_sequence = model_meta["input_sequence"]
    country = model_meta["country"]
    # get input for the model : last n values of percent renewable
    input_data = get_percent_actual_generation(country, input_sequence)
    print(input_data.shape)
    input_percentage = input_data["percentRenewable"].tolist()
    input_start = input_data.iloc[0]["startTimeUTC"]
    input_end = input_data.iloc[-1]["startTimeUTC"]

    # get the scaler
    scaler = get_scaler(model_meta)

    # run the model
    output = predict(model_name, input_data, scaler, input_sequence)

    return {
        "input": {
            "country": country,
            "model": model_name,
            "percentRenewable": input_percentage,
            "start": input_start,
            "end": input_end
        },
        "output": output
    }
