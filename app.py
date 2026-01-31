from flask import Flask, request, render_template
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Initialize the Flask app
application = Flask(__name__)
app = application

# Route for the home page
@app.route('/')
def index():
    return render_template('home.html') 

# Route for prediction logic
@app.route('/predict', methods=['POST'])
def predict_datapoint():
    # 1. Capture data from the HTML form
    data = CustomData(
        type_val=request.form.get('type_val'),
        shipping_mode=request.form.get('shipping_mode'),
        region=request.form.get('region'),
        market=request.form.get('market'),
        sales=float(request.form.get('sales')),
        qty=int(request.form.get('qty')),
        total=float(request.form.get('total')),
        benefit=float(request.form.get('benefit')),
        discount=float(request.form.get('discount')),
        order_date=request.form.get('order_date'),
        ship_date=request.form.get('ship_date')
    )
    
    # 2. Convert captured data into a DataFrame
    pred_df = data.get_data_as_data_frame()
    
    # 3. Call the Predict Pipeline
    predict_pipeline = PredictPipeline()
    results = predict_pipeline.predict(pred_df)
    
    # 4. Return result back to the web page
    output = "LATE RISK" if results[0] == 1 else "ON TIME"
    return render_template('home.html', results=output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)