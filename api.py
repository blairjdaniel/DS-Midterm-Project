# import modules
from flask import request, jsonify, Flask
import traceback
import pandas as pd
import time
from housing_predictions import loadModel

# create Flask app object (note the __name__ tells it to run when the function "__name__" runs... which is our
# main function that itself runs when the script is run in a terminal)
app = Flask(__name__)

# load the model
model = loadModel('/Users/blairjdaniel/lighthouse/lighthouse/week_nine/DS-Midterm-Project/notebooks/best_model.pkl')

# simple welcome page
@app.route('/')
def welcome():
    '''
        This function runs when the user accesses the base url page (no specific endpoint). It will get
        the current time and show it in a message.
    '''
    timenow = time.localtime()
    return f"Hello, this is a Flask demo -- the time is {timenow.tm_hour:02d}:{timenow.tm_min:02d}:{timenow.tm_sec:02d}"

# used car selling price prediction end-point
@app.route('/predict_sell_price',methods=["GET","POST"])
def prediction():
    '''
        This function is mainly for running the model to predict on data sent to it through a post request but also has
        a get request functionality that just sends back a simple message.
    '''

    # check if the request type is GET and if so return a message
    if request.method == 'GET':
        return "no, don't get... use post!"
    
    # check if the request type is POST (i.e., data will be received in JSON format)
    if request.method == 'POST':
        # get the received data from the input request
        data = request.json

        # package the data as a dataframe (note we are assuming the JSON input data matches the structure
        # expected as a dictionary with keys that are the appropriate columns and their respective values)
        X = pd.json_normalize([data])

        # use a try-except structure to attempt our prediction
        try:
            # try getting the prediction from the model
            y = model.predict(X)

            # return a JSON-formatted dictionary with a simple message key (stating "success") as well
            # as the prediction(s) made by the model converted to a list
            return jsonify({'message': 'success', 'prediction': list(y)})
        
        # if the prediction process did not work for whatever reason, package the error message in a JSON dictionary
        except:
            return jsonify({
               "trace": traceback.format_exc()
               })

# main function (runs when the script is run in a terminal using "python app.py")
if __name__ == "__main__":
    # here we set the host to 0.0.0.0 so that it will be locally available (http://localhost) as well as over
    # any local network we are connected to such as a home WiFi network (you would need to find your IP address
    # on the network and can do that by running "ipconfig" in a Windows terminal -- it should be something like
    # 192.168.0.4)
    app.run(host="0.0.0.0",port=5555)