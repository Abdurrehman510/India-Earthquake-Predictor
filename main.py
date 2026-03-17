import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import distance
from geopy.geocoders import Nominatim
import osmnx as ox
import folium
import folium.plugins
import math
import requests
import json
from fuzzywuzzy import process

# Initialize session state for map and predictions
if 'map_html' not in st.session_state:
    st.session_state.map_html = None
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
if 'location_info' not in st.session_state:
    st.session_state.location_info = None

# Load cities from JSON
try:
    with open('assets/city_coordinates.json', 'r') as f:
        cities = json.load(f)
except FileNotFoundError:
    st.error("assets/city_coordinates.json file not found. Please ensure the file is in the correct directory.")
    st.stop()

# Function to calculate bearing (angle) between two points
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return bearing if bearing >= 0 else bearing + 360

# Function to calculate MMI using GMPE
def calculate_mmi(magnitude, distance_km):
    mmi = 1.68 * magnitude - 3.29 - 0.0206 * distance_km
    # Keep within MMI scale range 1–12
    return max(1, min(mmi, 12))


# Function to calculate Effective Magnitude
def calculate_effective_magnitude(magnitude, distance_km):
    effective_mag = magnitude - (4/3) * np.log10(distance_km + 1)
    return max(1.0, min(effective_mag, 10.0))  # Clamp between 1.0 and 10.0

# Function to get coordinates from location string (name or lat,lon)
def get_coordinates(location_str):
    if location_str in cities:
        return cities[location_str][0], cities[location_str][1]
    city_names = list(cities.keys())
    match = process.extractOne(location_str, city_names, score_cutoff=80)
    if match:
        matched_city = match[0]
        st.info(f"Did you mean '{matched_city}'? Using coordinates for {matched_city}.")
        return cities[matched_city][0], cities[matched_city][1]
    try:
        lat, lon = map(float, location_str.split(','))
        return lat, lon
    except ValueError:
        geolocator = Nominatim(user_agent="earthquake_app")
        location = geolocator.geocode(location_str + ", India")
        if location:
            return location.latitude, location.longitude
        st.error(f"Could not find location '{location_str}'. Please check spelling or use coordinates.")
        return None, None

# Function to get live location using ipinfo.io
def get_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        response.raise_for_status()
        data = response.json()
        location = data['loc'].split(',')
        return {
            'city': data.get('city', 'Unknown'),
            'region': data.get('region', 'Unknown'),
            'country': data.get('country', 'Unknown'),
            'latitude': float(location[0]),
            'longitude': float(location[1])
        }
    except Exception as e:
        st.warning(f"Could not fetch live location: {e}")
        return None

# Load preprocessed data and model at startup
try:
    df = pd.read_csv("assets/cleaned_india_earthquakes.csv")
    average_depth = df['depth'].mean()
    st.write(f"Average earthquake depth used: {average_depth:.2f} km")
except FileNotFoundError:
    st.error("Cleaned data not found. Please ensure 'assets/cleaned_india_earthquakes.csv' is available.")
    st.stop()

try:
    model = joblib.load("assets/earthquake_model.pkl")
    print(type(model))  # Debug: Verify model type
except FileNotFoundError:
    st.error("Trained model not found. Please ensure 'assets/earthquake_model.pkl' is available.")
    st.stop()

# Streamlit app
st.title("Earthquake Impact Prediction for India")
st.write("Enter a location within India or use your live location to predict earthquake impact and get safety recommendations. Today's date and time: 04:25 PM IST, August 11, 2025.")

# Live location button
st.write("Click below to get your current location details and coordinates:")
if st.button("Get Live Location"):
    st.session_state.location_info = get_location()
    if st.session_state.location_info:
        info = st.session_state.location_info
        st.write(f"**City**: {info['city']}")
        st.write(f"**Region**: {info['region']}")
        st.write(f"**Country**: {info['country']}")
        st.write(f"**Coordinates**: Latitude {info['latitude']}, Longitude {info['longitude']}")
        st.write("Copy the coordinates above into the input fields as needed.")
    else:
        st.error("Failed to fetch live location. Please enter coordinates manually.")

# Input fields
epicenter_location = st.text_input("Enter Epicenter Location (e.g., Delhi or lat,lon)", "Delhi")
user_location = st.text_input("Enter Your Location (e.g., Mumbai or lat,lon)", "")
magnitude = st.number_input("Enter Earthquake Magnitude", min_value=1.0, max_value=10.0, value=6.0, step=0.1)

if st.button("Predict"):
    # Reset session state for new prediction
    st.session_state.predictions_made = True
    st.session_state.map_html = None

    # Geocode epicenter and user locations
    epicenter_lat, epicenter_lon = get_coordinates(epicenter_location)
    user_lat, user_lon = get_coordinates(user_location)

    if not epicenter_lat or not epicenter_lon:
        st.error("Invalid epicenter location. Please enter a valid place name or coordinates (e.g., 'Delhi' or '28.6139,77.209').")
        st.stop()
    if not user_lat or not user_lon:
        st.error("Invalid user location. Please enter a valid place name or coordinates (e.g., 'Mumbai' or '19.076,72.8777').")
        st.stop()

    # Create map centered on epicenter
    m = folium.Map(location=[epicenter_lat, epicenter_lon], zoom_start=5, width=675, height=400)
    folium.Marker([epicenter_lat, epicenter_lon], popup=f"Epicenter\nMagnitude: {magnitude}", icon=folium.Icon(color="red")).add_to(m)
    folium.plugins.Fullscreen(
        position='topright',
        title='Expand to full screen',
        title_cancel='Exit full screen'
    ).add_to(m)

    # Predict for major cities
    danger_zones = []
    safe_zones = []
    feature_names = ['magnitude', 'distance', 'angle', 'depth']

    for city, (city_lat, city_lon) in cities.items():
        dist = distance((epicenter_lat, epicenter_lon), (city_lat, city_lon)).km
        angle = calculate_bearing(epicenter_lat, epicenter_lon, city_lat, city_lon)
        X_city = pd.DataFrame([[magnitude, dist, angle, average_depth]], columns=feature_names)
        prob = model.predict_proba(X_city)[0, 1]
        mmi = calculate_mmi(magnitude, dist)
        effective_mag = calculate_effective_magnitude(magnitude, dist)
        city_info = {
            "name": city,
            "distance": dist,
            "probability": prob,
            "mmi": mmi,
            "effective_magnitude": effective_mag
        }
        if prob >= 0.5 and effective_mag>=5:
            danger_zones.append(city_info)
            folium.Marker(
                [city_lat, city_lon],
                popup=f"{city}\nDistance: {dist:.1f} km\nProb: {prob:.2f}\nMMI: {mmi:.1f}\nEffective Mag: {effective_mag:.1f}",
                icon=folium.Icon(color="red")
            ).add_to(m)
        elif prob > 0.01 and effective_mag>= 2.5 and len(safe_zones)<25 and dist<100:  # Exclude probabilities near 0.0
            safe_zones.append(city_info)
            folium.Marker(
                [city_lat, city_lon],
                popup=f"{city}\nDistance: {dist:.1f} km\nProb: {prob:.2f}\nMMI: {mmi:.1f}\nEffective Mag: {effective_mag:.1f}",
                icon=folium.Icon(color="green")
            ).add_to(m)

    # Sort zones by distance
    danger_zones = sorted(danger_zones, key=lambda x: x["distance"])
    safe_zones = sorted(safe_zones, key=lambda x: x["distance"])

    # User's safety information
    dist_user = distance((epicenter_lat, epicenter_lon), (user_lat, user_lon)).km
    angle_user = calculate_bearing(epicenter_lat, epicenter_lon, user_lat, user_lon)
    X_user = pd.DataFrame([[magnitude, dist_user, angle_user, average_depth]], columns=feature_names)
    prob_user = model.predict_proba(X_user)[0, 1]  # Fixed: Use X_user instead of X_city
    mmi_user = calculate_mmi(magnitude, dist_user)
    effective_mag_user = calculate_effective_magnitude(magnitude, dist_user)
    folium.Marker(
        [user_lat, user_lon],
        popup=f"Your Location\nDistance: {dist_user:.1f} km\nProb: {prob_user:.2f}\nMMI: {mmi_user:.1f}\nEffective Mag: {effective_mag_user:.1f}",
        icon=folium.Icon(color="blue")
    ).add_to(m)

    # Fetch top 20 hospitals and grounds near user's location
    try:
        hospitals = ox.features_from_point((user_lat, user_lon), tags={"amenity": "hospital"}, dist=20000)
        hospitals_list = []
        for _, hospital in hospitals.iterrows():
            if "name" in hospital and hospital["name"]:
                h_lat = hospital.geometry.centroid.y
                h_lon = hospital.geometry.centroid.x
                h_dist = distance((user_lat, user_lon), (h_lat, h_lon)).km
                hospitals_list.append({"name": hospital["name"], "distance": h_dist, "lat": h_lat, "lon": h_lon})
        hospitals_list = sorted(hospitals_list, key=lambda x: x["distance"])[:20]
        for h in hospitals_list:
            folium.Marker(
                [h["lat"], h["lon"]],
                popup=f"Hospital: {h['name']}\nDistance: {h['distance']:.1f} km",
                icon=folium.Icon(color="red", icon="plus")
            ).add_to(m)
    except Exception as e:
        st.warning(f"Could not fetch hospitals: {e}")
        hospitals_list = []

    try:
        grounds = ox.features_from_point((user_lat, user_lon), tags={"leisure": ["park", "recreation_ground"]}, dist=20000)
        grounds_list = []
        for _, ground in grounds.iterrows():
            if "name" in ground and ground["name"]:
                g_lat = ground.geometry.centroid.y
                g_lon = ground.geometry.centroid.x
                g_dist = distance((user_lat, user_lon), (g_lat, g_lon)).km
                grounds_list.append({"name": ground["name"], "distance": g_dist, "lat": g_lat, "lon": g_lon})
        grounds_list = sorted(grounds_list, key=lambda x: x["distance"])[:20]
        for g in grounds_list:
            folium.Marker(
                [g["lat"], g["lon"]],
                popup=f"Ground: {g['name']}\nDistance: {g['distance']:.1f} km",
                icon=folium.Icon(color="green", icon="leaf")
            ).add_to(m)
    except Exception as e:
        st.warning(f"Could not fetch grounds: {e}")
        grounds_list = []

    # Save map HTML to session state
    st.session_state.map_html = m._repr_html_()

# Display map if predictions have been made
if st.session_state.predictions_made and st.session_state.map_html:
    st.subheader("Earthquake Impact Map")
    st.write("Red marker: Epicenter | Red pins: Danger zones | Green pins: Safe zones | Blue pin: Your location | Red plus: Hospitals | Green leaf: Grounds")
    st.write("Click the button in the top-right corner of the map to toggle full-screen mode.")
    st.components.v1.html(st.session_state.map_html, width=750, height=600, scrolling=False)

# Display safety recommendations if predictions have been made
if st.session_state.predictions_made:
    st.subheader("Safety Recommendations")
    st.write(f"**Your Location Risk (Distance: {dist_user:.1f} km):**")
    st.write(f"- Probability of strong shaking (MMI ≥ 6): {prob_user:.2f}")
    st.write(f"- Estimated MMI at your location: {mmi_user:.1f}")
    st.write(f"- Effective Magnitude at your location: {effective_mag_user:.1f}")
    st.write("*Effective Magnitude is the hypothetical magnitude of an earthquake centered at this location that would cause the same shaking intensity (MMI) as the actual earthquake.*")
    if prob_user > 0.8 or mmi_user >= 7.0:
        st.write("- **High Risk**: Immediate action required. Drop, cover, and hold on under sturdy furniture. Evacuate to an open ground if safe.")
    elif prob_user > 0.5 or mmi_user >= 5.0:
        st.write("- **Moderate Risk**: Prepare for shaking. Secure loose objects and stay away from windows. Consider moving to a safe zone.")
    else:
        st.write("- **Low Risk**: Stay alert. Follow general safety guidelines and monitor for aftershocks.")

    # Estimated seismic wave arrival times
    p_wave_time = dist_user / 6  # P-wave speed ~6 km/s
    s_wave_time = dist_user / 3.5  # S-wave speed ~3.5 km/s
    st.write("**Estimated Seismic Wave Arrival Times:**")
    st.write(f"- Primary (P) Wave: ~{p_wave_time:.1f} seconds")
    st.write(f"- Secondary (S) Wave: ~{s_wave_time:.1f} seconds")

    # Nearby cities
    if danger_zones:
        st.write("**Danger Zones (High Risk, Probability ≥ 0.5):**")
        for zone in danger_zones:
            st.write(f"- {zone['name']}: {zone['distance']:.1f} km away, Probability {zone['probability']:.2f}, MMI {zone['mmi']:.1f}, Effective Mag {zone['effective_magnitude']:.1f}")
    else:
        st.write("No nearby cities are in the danger zone.")

    if safe_zones:
        st.write("**Safe Zones (Low Risk, Probability < 0.5):**")
        for zone in safe_zones:
            st.write(f"- {zone['name']}: {zone['distance']:.1f} km away, Probability {zone['probability']:.2f}, MMI {zone['mmi']:.1f}, Effective Mag {zone['effective_magnitude']:.1f}")
    else:
        st.write("No nearby cities are in the safe zone.")

    # Nearby hospitals and grounds
    if hospitals_list:
        st.write("**Top 20 Nearest Hospitals (within 20 km):**")
        for h in hospitals_list:
            st.write(f"- {h['name']}: {h['distance']:.1f} km away")
    else:
        st.write("No hospitals found within 20 km. Seek medical attention at the nearest city.")

    if grounds_list:
        st.write("**Top 20 Nearest Open Grounds for Safety (within 20 km):**")
        for g in grounds_list:
            st.write(f"- {g['name']}: {g['distance']:.1f} km away")
    else:
        st.write("No open grounds found within 20 km. Proceed to the nearest safe zone.")

    # Additional safety tips
    st.write("**General Safety Tips:**")
    st.write("- **Before**: Secure heavy objects, prepare an emergency kit (water, food, first aid, flashlight).")
    st.write("- **During**: Drop, cover, and hold on. Avoid windows and heavy fixtures.")
    st.write("- **After**: Check for injuries, avoid damaged areas, and prepare for aftershocks.")
    st.write("- **Emergency Contacts**: Dial 100 (Police), 108 (Ambulance) in India.")