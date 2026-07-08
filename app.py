from flask import Flask,request,render_template
import numpy as np
import pandas as pd
import threading
import json
import os

from sklearn.preprocessing import StandardScaler
from src.pipelines.predict_pipeline import CustomData,PredictPipeline
from src.pipelines.train_pipeline import TrainPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('race_ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])

@app.route('/train', methods=['GET'])
def train_model():
    def run_training():
        train_pipeline = TrainPipeline()
        score = train_pipeline.run_pipeline()
        print(f"Training completed successfully. R2 score: {score}")

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    return render_template('index.html', train_message='Training started in the background. Check the terminal for completion.')
    
@app.route('/analytics')
def analytics():
    data_path = os.path.join('artifacts', 'data.csv')
    if not os.path.exists(data_path):
        return render_template('analytics.html', error="Data file not found.")
        
    df = pd.read_csv(data_path)
    raw_data = df.to_dict('records')
    
    return render_template('analytics.html', raw_data=json.dumps(raw_data))

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        