import pickle
from flask import Flask,request,jsonify,render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)


# Load the trained model
ridge_model=pickle.load(open('C:\\Users\\ASUS\\Desktop\\Model\\Forest_Fire_END_TO_END_PROJECT\\Models\\forest_fire_Ml_Model_ridge_pickle_file','rb'))

standard_scaler=pickle.load(open('Forest_Fire_END_TO_END_PROJECT/Models/forest_fire_scaler_pickle_file','rb'))

# route for home page
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='POST':
        # store the data from the form
        Temperature=float(request.form.get('Temperature'))
        RH=float(request.form.get('RH'))
        Ws=float(request.form.get('Ws'))
        Rain=float(request.form.get('Rain'))
        FFMC=float(request.form.get('FFMC'))
        DMC=float(request.form.get('DMC'))
        ISI=float(request.form.get('ISI'))
        Classes=float(request.form.get('Classes'))
        Region=float(request.form.get('Region'))

        # new data scaled
        new_data_scaled=standard_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])

        # predict the output using the loaded model
        result=ridge_model.predict(new_data_scaled)

        return render_template('home.html',result=result[0])

    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0")
