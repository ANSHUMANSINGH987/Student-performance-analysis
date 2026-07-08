from flask import Flask, request, render_template, jsonify
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
        
        input_data = pred_df.to_dict('records')[0]
        return render_template('home.html',results=results[0], input_data=input_data)

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    if request.method == 'GET':
        return render_template('train.html')
    
    # POST request to start training
    metrics_path = os.path.join('artifacts', 'model_metrics.json')
    if os.path.exists(metrics_path):
        os.remove(metrics_path)
        
    def run_training():
        try:
            train_pipeline = TrainPipeline()
            metrics = train_pipeline.run_pipeline()
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f)
            print("Training completed and metrics saved.")
        except Exception as e:
            with open(metrics_path, 'w') as f:
                json.dump({"error": str(e)}, f)
            print(f"Training failed: {e}")

    training_thread = threading.Thread(target=run_training, daemon=True)
    training_thread.start()
    return jsonify({"status": "started"})

@app.route('/training_status', methods=['GET'])
def training_status():
    metrics_path = os.path.join('artifacts', 'model_metrics.json')
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            if "error" in metrics:
                return jsonify({"status": "error", "error": metrics["error"]})
            return jsonify({"status": "completed", "metrics": metrics})
        except Exception:
            return jsonify({"status": "training"})
    else:
        return jsonify({"status": "training"})
    
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