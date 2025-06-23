import streamlit as st
import requests
from geopy.geocoders import Nominatim
import pickle
import numpy as np

# === CONFIG ===
OPENWEATHER_API_KEY = "56023c1ff6bc1c084c05e0016367bfd0"  # Replace with your key

# === FUNCTION: Get Coordinates ===
def get_coordinates(address):
    geolocator = Nominatim(user_agent="crop_predictor")
    location = geolocator.geocode(address)
    if location:
        return location.latitude, location.longitude
    else:
        return None, None

# === FUNCTION: Get Weather Data ===
def get_weather_data(lat, lon):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        res = requests.get(url)
        data = res.json()
        temp = data['main']['temp']
        humidity = data['main']['humidity']
        rainfall = data.get('rain', {}).get('1h', 0.0)  # may not be present
        return temp, humidity, rainfall
    except:
        return None, None, None

# === FUNCTION: Get Soil pH ===
def get_soil_ph(lat, lon):
    url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&value=mean"
    try:
        res = requests.get(url)
        data = res.json()
        layers = data.get('properties', {}).get('phh2o', {}).get('layers', [])
        if layers:
            ph_value = layers[0]['depths'][0]['values']['mean']
            return round(ph_value / 10, 1)
        else:
            return None
    except Exception as e:
        return None
        
# === Load Trained Model ===
@st.cache_resource
def load_model():
    with open("RandomForest.pkl", "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# === STREAMLIT UI ===
st.title("🌾 Crop Recommendation App")
address = st.text_input("Enter your farm location:")

if address:
    with st.spinner("Getting coordinates..."):
        lat, lon = get_coordinates(address)
    
    if lat is not None:
        st.success(f"📍 Latitude: {lat:.4f}, Longitude: {lon:.4f}")

        with st.spinner("Fetching weather and soil data..."):
            temp, humidity, rainfall = get_weather_data(lat, lon)
            ph = get_soil_ph(lat, lon)

        if None in [temp, humidity, ph, rainfall]:
            st.error("❌ Unable to retrieve all required data. Try a different address.")
        else:
            st.subheader("📊 Retrieved Environmental Data:")
            st.write(f"🌡️ Temperature: {temp} °C")
            st.write(f"💧 Humidity: {humidity} %")
            st.write(f"🌧️ Rainfall: {rainfall} mm")
            st.write(f"🧪 Soil pH: {ph}")

            # === Prepare Input for Model ===
            input_features = np.array([[temp, humidity, ph, rainfall]]) 
            prediction = model.predict(input_features)[0]
            st.success(f"✅ Recommended Crop: **{prediction}**")
    else:
        st.error("⚠️ Could not find that location.")
