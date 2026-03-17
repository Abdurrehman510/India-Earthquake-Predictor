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
from django.shortcuts import render
from .forms import PredictionForm
from django.utils import timezone

# Load cities from JSON
try:
    with open('assets/city_coordinates.json', 'r') as f:
        cities = json.load(f)
except FileNotFoundError:
    cities = {}  # Handle error in template

# Functions from main.py (preserved)
def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    bearing = math.degrees(math.atan2(x, y))
    return bearing if bearing >= 0 else bearing + 360

def calculate_mmi(magnitude, distance_km):
    mmi = 1.68 * magnitude - 3.29 - 0.0206 * distance_km
    return max(1, min(mmi, 12))

def calculate_effective_magnitude(magnitude, distance_km):
    effective_mag = magnitude - (4/3) * np.log10(distance_km + 1)
    return max(1.0, min(effective_mag, 10.0))

def get_coordinates(location_str):
    if location_str in cities:
        return cities[location_str][0], cities[location_str][1]
    city_names = list(cities.keys())
    match = process.extractOne(location_str, city_names, score_cutoff=80)
    if match:
        matched_city = match[0]
        return cities[matched_city][0], cities[matched_city][1]
    try:
        lat, lon = map(float, location_str.split(','))
        return lat, lon
    except ValueError:
        geolocator = Nominatim(user_agent="earthquake_app")
        location = geolocator.geocode(location_str + ", India")
        if location:
            return location.latitude, location.longitude
        return None, None

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
    except Exception:
        return None

# Load data and model (cached at app level)
try:
    df = pd.read_csv("assets/cleaned_india_earthquakes.csv")
    average_depth = df['depth'].mean() if not df.empty else 0
    model = joblib.load("assets/earthquake_model.pkl")
except Exception as e:
    df = None
    average_depth = 0
    model = None

def index(request):
    error = None
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            epicenter_location = form.cleaned_data['epicenter_location']
            user_location_input = form.cleaned_data['user_location']
            magnitude = form.cleaned_data['magnitude']
            use_live = form.cleaned_data['use_live_location']

            if use_live:
                location_info = get_location()
                if location_info:
                    user_lat = location_info['latitude']
                    user_lon = location_info['longitude']
                    user_location = f"{location_info['city']}, {location_info['region']}"
                else:
                    error = "Failed to fetch live location. Please enter manually."
                    return render(request, 'predictor/index.html', {'form': form, 'error': error})
            else:
                user_location = user_location_input

            epicenter_lat, epicenter_lon = get_coordinates(epicenter_location)
            if not epicenter_lat:
                error = f"Invalid epicenter location '{epicenter_location}'."
                return render(request, 'predictor/index.html', {'form': form, 'error': error})

            user_lat, user_lon = get_coordinates(user_location)
            if not user_lat:
                error = f"Invalid user location '{user_location}'."
                return render(request, 'predictor/index.html', {'form': form, 'error': error})

            if model is None or df is None or df.empty:
                error = "Model or data not loaded. Run training commands first."
                return render(request, 'predictor/index.html', {'form': form, 'error': error})

            m = folium.Map(location=[epicenter_lat, epicenter_lon], zoom_start=5)
            folium.Marker([epicenter_lat, epicenter_lon], popup=f"Epicenter\nMagnitude: {magnitude}", icon=folium.Icon(color="red")).add_to(m)
            folium.plugins.Fullscreen(position='topright', title='Expand to full screen', title_cancel='Exit full screen').add_to(m)

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
                city_info = {"name": city, "distance": dist, "probability": prob, "mmi": mmi, "effective_magnitude": effective_mag}
                if prob >= 0.5 and effective_mag >= 5:
                    danger_zones.append(city_info)
                    folium.Marker([city_lat, city_lon], popup=f"{city}\nDistance: {dist:.1f} km\nProb: {prob:.2f}\nMMI: {mmi:.1f}\nEffective Mag: {effective_mag:.1f}", icon=folium.Icon(color="red")).add_to(m)
                elif prob > 0.01 and effective_mag >= 2.5 and len(safe_zones) < 25 and dist < 100:
                    safe_zones.append(city_info)
                    folium.Marker([city_lat, city_lon], popup=f"{city}\nDistance: {dist:.1f} km\nProb: {prob:.2f}\nMMI: {mmi:.1f}\nEffective Mag: {effective_mag:.1f}", icon=folium.Icon(color="green")).add_to(m)

            danger_zones = sorted(danger_zones, key=lambda x: x["distance"])
            safe_zones = sorted(safe_zones, key=lambda x: x["distance"])

            dist_user = distance((epicenter_lat, epicenter_lon), (user_lat, user_lon)).km
            angle_user = calculate_bearing(epicenter_lat, epicenter_lon, user_lat, user_lon)
            X_user = pd.DataFrame([[magnitude, dist_user, angle_user, average_depth]], columns=feature_names)
            prob_user = model.predict_proba(X_user)[0, 1]
            mmi_user = calculate_mmi(magnitude, dist_user)
            effective_mag_user = calculate_effective_magnitude(magnitude, dist_user)
            folium.Marker([user_lat, user_lon], popup=f"Your Location\nDistance: {dist_user:.1f} km\nProb: {prob_user:.2f}\nMMI: {mmi_user:.1f}\nEffective Mag: {effective_mag_user:.1f}", icon=folium.Icon(color="blue")).add_to(m)

            hospitals_list = []
            try:
                hospitals = ox.features_from_point((user_lat, user_lon), tags={"amenity": "hospital"}, dist=20000)
                for _, hospital in hospitals.iterrows():
                    if "name" in hospital and hospital["name"]:
                        h_lat = hospital.geometry.centroid.y
                        h_lon = hospital.geometry.centroid.x
                        h_dist = distance((user_lat, user_lon), (h_lat, h_lon)).km
                        hospitals_list.append({"name": hospital["name"], "distance": h_dist, "lat": h_lat, "lon": h_lon})
                hospitals_list = sorted(hospitals_list, key=lambda x: x["distance"])[:20]
                for h in hospitals_list:
                    folium.Marker([h["lat"], h["lon"]], popup=f"Hospital: {h['name']}\nDistance: {h['distance']:.1f} km", icon=folium.Icon(color="red", icon="plus")).add_to(m)
            except:
                pass

            grounds_list = []
            try:
                grounds = ox.features_from_point((user_lat, user_lon), tags={"leisure": ["park", "recreation_ground"]}, dist=20000)
                for _, ground in grounds.iterrows():
                    if "name" in ground and ground["name"]:
                        g_lat = ground.geometry.centroid.y
                        g_lon = ground.geometry.centroid.x
                        g_dist = distance((user_lat, user_lon), (g_lat, g_lon)).km
                        grounds_list.append({"name": ground["name"], "distance": g_dist, "lat": g_lat, "lon": g_lon})
                grounds_list = sorted(grounds_list, key=lambda x: x["distance"])[:20]
                for g in grounds_list:
                    folium.Marker([g["lat"], g["lon"]], popup=f"Ground: {g['name']}\nDistance: {g['distance']:.1f} km", icon=folium.Icon(color="green", icon="leaf")).add_to(m)
            except:
                pass

            map_html = m._repr_html_()

            p_wave_time = dist_user / 6
            s_wave_time = dist_user / 3.5

            context = {
                'form': form,
                'map_html': map_html,
                'predictions_made': True,
                'dist_user': dist_user,
                'prob_user': prob_user,
                'mmi_user': mmi_user,
                'effective_mag_user': effective_mag_user,
                'p_wave_time': p_wave_time,
                's_wave_time': s_wave_time,
                'danger_zones': danger_zones,
                'safe_zones': safe_zones,
                'hospitals_list': hospitals_list,
                'grounds_list': grounds_list,
                'average_depth': average_depth,
                'current_datetime': timezone.now(),
            }
            return render(request, 'predictor/index.html', context)
    else:
        form = PredictionForm()

    return render(request, 'predictor/index.html', {'form': form, 'error': error})

# predictor/views.py

def compute_risk_context(prob_user, mmi_user, effective_mag_user, dist_user):
    """
    Compute risk level and presentation data for template.
    Returns a dict ready to include in template context.
    """
    # Thresholds (inclusive on boundaries)
    HIGH_PROB = 0.8
    MED_PROB = 0.5
    HIGH_MMI = 7.0
    MED_MMI = 5.0

    if prob_user >= HIGH_PROB or mmi_user >= HIGH_MMI:
        level = "high"
        indicator = "High Risk"
        icon = "fire"
        icon_box = "icon-box-danger"
        alert_class = "alert-danger"
        alert_icon = "fire"
        badge_level_class = "risk-high"
        recommendation = (
            "Immediate action required. Drop, cover, and hold on under sturdy furniture. "
            "Evacuate to an open ground if safe."
        )
    elif prob_user >= MED_PROB or mmi_user >= MED_MMI:
        level = "medium"
        indicator = "Moderate Risk"
        icon = "exclamation"
        icon_box = "icon-box-warning"
        alert_class = "alert-warning"
        alert_icon = "exclamation-triangle"
        badge_level_class = "risk-medium"
        recommendation = (
            "Prepare for shaking. Secure loose objects and stay away from windows. "
            "Consider moving to a safe zone."
        )
    else:
        level = "low"
        indicator = "Low Risk"
        icon = "check-circle"
        icon_box = "icon-box-success"
        alert_class = "alert-success"
        alert_icon = "shield-alt"
        badge_level_class = "risk-low"
        recommendation = "Stay alert. Follow general safety guidelines and monitor for aftershocks."

    return {
        "risk_level": level,
        "indicator": indicator,
        "icon": icon,
        "icon_box": icon_box,
        "alert_class": alert_class,
        "alert_icon": alert_icon,
        "badge_level_class": badge_level_class,
        "recommendation": recommendation,
        # keep these original numeric values handy for display
        "dist_user": dist_user,
        "mmi_user": mmi_user,
        "prob_user": prob_user,
        "effective_mag_user": effective_mag_user,
    }
