# 🌍 India Earthquake Predictor

![Project Preview](preview.png)

A professional, data-driven application designed to predict the impact of earthquakes in India and provide immediate safety recommendations. This project features both a **Django** web application and a **Streamlit** dashboard for versatile usage.

## 🚀 Features

-   **Real-time Risk Assessment**: Predicts shaking intensity (MMI) and effective magnitude based on location and earthquake parameters.
-   **Interactive Mapping**: Visualises epicenter, danger zones, and safe zones using Folium.
-   **Live Location Integration**: Automatically detects user location for localized risk analysis.
-   **Safety Resources**: Dynamically fetches nearby hospitals and open grounds using OpenStreetMap data (OSMNX).
-   **Early Warning Estimates**: Calculates P-wave and S-wave arrival times for immediate awareness.
-   **Dual Interface**: Supports both a robust Django backend and a lightweight Streamlit frontend.

## 🛠️ Installation & Setup

### Prerequisites
-   Python 3.8+
-   Virtual Environment (recommended)

### Steps
1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Abdurrehman510/India-Earthquake-Predictor.git
    cd India-Earthquake-Predictor
    ```

2.  **Set Up Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Running the Application

### Option 1: Django Web App
```bash
cd india_earthquake_predictor
python manage.py runserver
```

### Option 2: Streamlit Dashboard
```bash
streamlit run main.py
```

## 📊 Model & Data
The application utilizes a machine learning model trained on historical earthquake data in the Indian subcontinent. It considers magnitude, distance, bearing, and depth to estimate impact probabilities.

---
**Author**: Abdurrehman510
**License**: MIT
