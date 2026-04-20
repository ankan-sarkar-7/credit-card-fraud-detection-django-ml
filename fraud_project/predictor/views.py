import os
from django.shortcuts import render
import pandas as pd
import joblib
from geopy.distance import geodesic

# Load model safely
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = joblib.load(os.path.join(BASE_DIR, "predictor", "fraud_detection_model.jb"))
encoder = joblib.load(os.path.join(BASE_DIR, "predictor", "label_encoder.jb"))

def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

def home(request):
    return render(request, "index.html")

def predict(request):
    if request.method == "POST":
        try:
            # Get values
            merchant = request.POST.get("merchant")
            category = request.POST.get("category")
            amt = float(request.POST.get("amt"))

            lat = float(request.POST.get("lat"))
            long = float(request.POST.get("long"))
            merch_lat = float(request.POST.get("merch_lat"))
            merch_long = float(request.POST.get("merch_long"))

            hour = int(request.POST.get("hour"))
            day = int(request.POST.get("day"))
            month = int(request.POST.get("month"))

            gender = request.POST.get("gender")
            cc_num = request.POST.get("cc_num")

            # ✅ Coordinate validation
            if not (-90 <= lat <= 90 and -90 <= merch_lat <= 90):
                return render(request, "result.html", {"result": "Invalid Latitude ❌"})

            if not (-180 <= long <= 180 and -180 <= merch_long <= 180):
                return render(request, "result.html", {"result": "Invalid Longitude ❌"})

            # Distance calculation
            distance = haversine(lat, long, merch_lat, merch_long)

            # Create DataFrame
            input_data = pd.DataFrame(
                [[merchant, category, amt, distance, hour, day, month, gender, cc_num]],
                columns=['merchant', 'category', 'amt', 'distance', 'hour', 'day', 'month', 'gender', 'cc_num']
            )

            # Encode categorical
            categorical_col = ['merchant', 'category', 'gender']

            for col in categorical_col:
                try:
                    input_data[col] = encoder[col].transform(input_data[col])
                except:
                    input_data[col] = -1

            # Hash credit card
            input_data['cc_num'] = input_data['cc_num'].apply(lambda x: hash(x) % (10 ** 2))

            # Prediction
            prediction = model.predict(input_data)[0]

            result = "🚨 Fraudulent Transaction" if prediction == 1 else "✅ Legitimate Transaction"

            return render(request, "result.html", {"result": result})

        except Exception as e:
            return render(request, "result.html", {"result": "Invalid Input ❌"})

    return render(request, "index.html")