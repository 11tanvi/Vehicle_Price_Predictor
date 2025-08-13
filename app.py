# from flask import Flask, render_template, request
# import joblib
# import pandas as pd
# import numpy as np
#
# app = Flask(__name__)
#
# # Load saved model and encoders
# model = joblib.load('model.pkl')
# encoders = joblib.load('encoders.pkl')
#
# feature_order = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'car_age', 'mileage_per_year']
#
# # Get allowed categories from encoders for validation
# allowed_categories = {col: list(le.classes_) for col, le in encoders.items()}
#
# @app.route('/')
# def home():
#     return render_template('index.html', allowed=allowed_categories)
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Get form data
#         brand = request.form['brand']
#         fuel = request.form['fuel']
#         seller_type = request.form['seller_type']
#         transmission = request.form['transmission']
#         owner = request.form['owner']
#         car_age = int(request.form['car_age'])
#         mileage_per_year = float(request.form['mileage_per_year'])
#
#         # Validate categorical inputs
#         for col, val in [('brand', brand), ('fuel', fuel), ('seller_type', seller_type),
#                          ('transmission', transmission), ('owner', owner)]:
#             if val not in allowed_categories[col]:
#                 return f"Invalid input for {col}: {val}. Allowed values: {allowed_categories[col]}"
#
#         # Create input dict
#         input_dict = {
#             'fuel': fuel,
#             'seller_type': seller_type,
#             'transmission': transmission,
#             'owner': owner,
#             'brand': brand,
#             'car_age': car_age,
#             'mileage_per_year': mileage_per_year
#         }
#
#         # Create DataFrame in correct order
#         input_df = pd.DataFrame([input_dict], columns=feature_order)
#
#         # Encode categorical variables
#         for col, le in encoders.items():
#             input_df[col] = le.transform(input_df[col])
#
#         # Debug prints (optional)
#         print("Encoded input:", input_df)
#
#         # Predict (model predicts log1p of price)
#         pred_log = model.predict(input_df)[0]
#
#         # Convert back from log scale to actual price
#         price_inr = np.expm1(pred_log)
#         price_inr = round(price_inr, 2)
#
#         # Format price string with INR symbol
#         price_str = f"₹ {price_inr:,}"  # adds comma as thousand separator
#
#         return render_template('result.html', prediction=price_str)
#
#     except Exception as e:
#         return f"Error: {str(e)}"
#
# if __name__ == '__main__':
#     app.run(debug=True)
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load saved model, encoders, and dataset globally
model = joblib.load('model.pkl')
encoders = joblib.load('encoders.pkl')

# Load full dataset for recommendations and charts
df = pd.read_csv('CAR DETAILS FROM CAR DEKHO.csv')

# Feature engineering on df to match model input features
import datetime
current_year = datetime.datetime.now().year
df['brand'] = df['name'].apply(lambda x: x.split()[0])
df['car_age'] = current_year - df['year']
df['mileage_per_year'] = df['km_driven'] / df['car_age'].replace(0, 1)
df.drop(['name', 'year', 'km_driven'], axis=1, inplace=True)

feature_order = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'car_age', 'mileage_per_year']

# Allowed categories for form validation
allowed_categories = {col: list(le.classes_) for col, le in encoders.items()}

@app.route('/')
def home():
    return render_template('index.html', allowed=allowed_categories)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        brand = request.form['brand']
        fuel = request.form['fuel']
        seller_type = request.form['seller_type']
        transmission = request.form['transmission']
        owner = request.form['owner']
        car_age = int(request.form['car_age'])
        mileage_per_year = float(request.form['mileage_per_year'])

        # Validate inputs
        for col, val in [('brand', brand), ('fuel', fuel), ('seller_type', seller_type),
                         ('transmission', transmission), ('owner', owner)]:
            if val not in allowed_categories[col]:
                return f"Invalid input for {col}: {val}. Allowed values: {allowed_categories[col]}"

        # Prepare input DataFrame in order
        input_dict = {
            'fuel': fuel,
            'seller_type': seller_type,
            'transmission': transmission,
            'owner': owner,
            'brand': brand,
            'car_age': car_age,
            'mileage_per_year': mileage_per_year
        }
        input_df = pd.DataFrame([input_dict], columns=feature_order)

        # Encode categorical features
        for col, le in encoders.items():
            input_df[col] = le.transform(input_df[col])

        # Predict log1p price, convert back
        pred_log = model.predict(input_df)[0]
        price_inr = np.expm1(pred_log)
        price_inr = round(price_inr, 2)

        # Prepare chart data: price distribution & mileage for selected brand
        brand_cars = df[df['brand'] == brand]

        price_data = brand_cars['selling_price'].tolist()
        mileage_data = brand_cars['mileage_per_year'].tolist()
        age_data = brand_cars['car_age'].tolist()

        # Similar cars recommendation: close car_age & mileage
        similar_cars = brand_cars[
            (brand_cars['car_age'].between(car_age-2, car_age+2)) &
            (brand_cars['mileage_per_year'].between(mileage_per_year*0.7, mileage_per_year*1.3))
        ].sort_values('selling_price').head(5)

        recommended = similar_cars[['brand', 'car_age', 'mileage_per_year', 'selling_price']].to_dict(orient='records')

        return render_template('result.html',
                               prediction=f"₹ {price_inr:,}",
                               price_data=price_data,
                               mileage_data=mileage_data,
                               age_data=age_data,
                               recommended=recommended,
                               brand=brand)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

