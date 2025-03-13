# this file demonstrates the same functionality we saw in used_cars_inference.ipynb but
# within a python script file (.py) instead of a Jupyter notebook

# import modules
import pickle
import numpy as np
import pandas as pd
import joblib

# unction to load the model
def loadModel(model_file):
    '''
        This function loads a model from a pickle file.

        inputs:
            model_file:     string indicating the file location
        outputs:
            model:          the trained model
    '''
    model = joblib.load(model_file)
    return model

# function to generate new data as a dataframe object
def createNewData(
        year_built,
        baths,
        garage,
        stories,
        beds,
        num_days,
        central_air,
        dishwasher,
        fireplace,
        basement,
        price_per_sqft,
        median_value_per_sqft,
        type_encoded,
        city_encoded):
    '''
        This function accepts each feature value as a separate argument and packages them into
        a dataframe.
        inputs:
           

        outputs:
            df_new:         data point packaged as a dataframe object
    '''

    # set up feature columns
    columns = [
        'description.year_built',
        'description.baths',
        'description.garage',
        'description.stories',
        'description.beds',
        'num_days',
        'central_air',
        'dishwasher',
        'fireplace',
        'basement',
        'price_per_sqft',
        'median_value_per_sqft',
        'description.type_encoded',
        'city_encoded']

    # construct array from inputs
    X_new = np.array([
        year_built,
        baths,
        garage,
        stories,
        beds,
        num_days,
        central_air,
        dishwasher,
        fireplace,
        basement,
        price_per_sqft,
        median_value_per_sqft,
        type_encoded,
        city_encoded
        ])
    
    # package data into a dataframe
    df_new = pd.DataFrame(data=X_new.reshape(1,-1),columns=columns)

    return df_new

# function to run the model and make a prediction
def runPrediction(model,X):
    '''
        This function runs the model to make a prediction.

        inputs:
            model:          trained model
            X:              dataframe or numpy array of input data

        outputs:
            y:              predicted selling price
    '''

    # make prediction
    y = model.predict(X)

    return y

# main function (runs when the file is run through a terminal) -- run in a terminal with "python used_cars_inference.py"
if __name__ == "__main__":

    # load the model from a pickle file
    model = loadModel('/Users/blairjdaniel/lighthouse/lighthouse/week_nine/DS-Midterm-Project/notebooks/best_model.pkl')

    # package new data
    X = createNewData(
        2019,
        2,
        3,
        1,
        2,
        56,
        1,
        1,
        0,
        0,
        200,
        0.8,
        43,
        2)

    # run the model to predict
    y = runPrediction(model,X)

    # print the results
    print(f"Predicted selling price: {y[0]}")