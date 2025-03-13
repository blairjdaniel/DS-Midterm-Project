# this is a Streamlit demo showing off just how easy it is to set up and run a Streamlit app to use our model in a
# more user-friendly way

# import modules
import streamlit as st
from housing_predictions import loadModel, createNewData, runPrediction

# define page title configuration (sets the name of the page as well as gives it a logo, note the wide layout)
st.set_page_config(
    page_title='Predict USA House Prices',
    page_icon=':smile:',
    layout='wide'
)

# write some introductory message
st.write('''
            # House Price Prediction
         
            Input the requested parameters to predict the selling price of the house :house:
         ''')

# example date input
date = st.date_input('Choose a date')

# set up inputs for each of our features for our model
# we'll do this with each input from left-to-right as a separate column (otherwise each one will be above/below each
# other and take up the whole width of the page)

# create a set of columns (7 in this case)
cols = st.columns(14)

# for each columns...

# access the column (note cols is a list that we can index)
with cols[0]:
    # create a user input (number type) that has a label ('Year') and a default value (2025), store whatever the
    # user enters in the output variable (year)
    year = st.number_input('Year',value=2025)

import streamlit as st

cols = st.columns(14)

with cols[0]:
    year_built = st.number_input('description.year_built', value=2019)
with cols[1]:
    baths = st.number_input('description.baths', value=2)
with cols[2]:
    garage = st.number_input('description.garage', value=3)
with cols[3]:
    stories = st.number_input('description.stories', value=1)
with cols[4]:
    beds = st.number_input('description.beds', value=2)
with cols[5]:
    num_days = st.number_input('num_days', value=56)
with cols[6]:
    central_air = st.number_input('central_air', value=1)
with cols[7]:
    dishwasher = st.number_input('dishwasher', value=1)
with cols[8]:
    fireplace = st.number_input('fireplace', value=0)
with cols[9]:
    basement = st.number_input('basement', value=0)
with cols[10]:
    price_per_sqft = st.number_input('price_per_sqft', value=200)
with cols[11]:
    median_value_per_sqft = st.number_input('median_value_per_sqft', value=0.8)
with cols[12]:
    type_encoded = st.number_input('description.type_encoded', value=43)
with cols[13]:
    city_encoded = st.number_input('city_encoded', value=2)

# load the model from the pickle file
model = loadModel('/Users/blairjdaniel/lighthouse/lighthouse/week_nine/DS-Midterm-Project/notebooks/best_model.pkl')

# package the inputs into the appropriate structure
X = createNewData(
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
)

# get our predicted selling price
y = runPrediction(model,X)

# display the prediction on our page
st.write(
        f":house: predicted selling price: ${y[0]:.2f}"
)