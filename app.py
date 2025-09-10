import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import traceback

st.write("✅ App startup successful.")

st.set_page_config(
    page_title="🌱 AI-Based Crop Cycle Planner",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-end CSS styling
st.markdown("""
<style>
body {
    background-color: #f5f8fa;
    font-family: 'Segoe UI', Tahoma, sans-serif;
    color: #2c3e50;
}
h1, h2, h3, h4 { color: #1f3c88; }
.main-header {
    text-align: center; background-color: #1f3c88;
    padding: 2rem; border-radius: 15px; color: white;
    font-weight: bold; box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.metric-container {
    background-color: #ffffff; padding: 1.5rem; border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.05); margin-bottom: 1.5rem;
    color: #2c3e50;
}
.metric-container h3 { color: #1f3c88; font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
.metric-container p { color: #2c3e50; font-size: 1.1rem; margin: 0.2rem 0; }
.recommendation-box {
    background-color: #e8f4f8; padding: 1.5rem; border-radius: 15px;
    border-left: 6px solid #1f3c88; margin-top: 1.5rem;
}
.stButton>button {
    background-color: #1f3c88; color: white; border-radius: 8px; padding: 0.6rem; font-weight: bold; border: none;
}
.stButton>button:hover { background-color: #3355aa; }
</style>
""", unsafe_allow_html=True)

def check_files_exist(file_list):
    missing = []
    for f in file_list:
        try:
            with open(f, 'rb'):
                pass
        except FileNotFoundError:
            missing.append(f)
    return missing

required_files = [
    'crop_recommendation_model.pkl', 'yield_prediction_model.pkl',
    'crop_encoder.pkl', 'crop_encoder_yield.pkl',
    'crop_prices.json', 'crop_season_mapping.json', 'reference_data.json'
]
missing_files = check_files_exist(required_files)
if missing_files:
    st.error(f"Missing files: {', '.join(missing_files)}")
    st.stop()

@st.cache_data
def load_models_and_data():
    try:
        crop_model = joblib.load('crop_recommendation_model.pkl')
        yield_model = joblib.load('yield_prediction_model.pkl')
        crop_encoder = joblib.load('crop_encoder.pkl')
        crop_encoder_yield = joblib.load('crop_encoder_yield.pkl')
        with open('crop_prices.json', 'r') as f:
            crop_prices = json.load(f)
        with open('crop_season_mapping.json', 'r') as f:
            crop_season_mapping = json.load(f)
        with open('reference_data.json', 'r') as f:
            reference_data = json.load(f)
        return crop_model, yield_model, crop_encoder, crop_encoder_yield, crop_prices, crop_season_mapping, reference_data
    except Exception as e:
        st.error("❌ Error loading models/data files!")
        st.error(traceback.format_exc())
        return None, None, None, None, None, None, None

crop_model, yield_model, crop_encoder, crop_encoder_yield, crop_prices, crop_season_mapping, reference_data = load_models_and_data()

st.markdown("""
<div class="main-header">
    <h1>🌾 AI-Based Crop Cycle Planner</h1>
    <p>Government-grade Multi-Season Crop Recommendation System</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("📋 Farm Info")
    state = st.selectbox("📍 Select State", reference_data.get('states', []) if reference_data else ["State1"])
    district = st.selectbox("📍 Select District", reference_data.get('districts', {}).get(state, []) if reference_data else ["District1"])
    area_hectares = st.number_input("🏞️ Farm Area (hectares)", 0.1, 1000.0, 2.0, step=0.1)

    st.header("🌱 Soil Parameters")
    soil_params = {
        'ph': st.slider("🌡️ Soil pH", 4.0, 10.0, 7.0, step=0.1),
        'nitrogen': st.slider("🌿 Nitrogen (kg/ha)", 0, 500, 250),
        'phosphorus': st.slider("💧 Phosphorus (kg/ha)", 0, 100, 50),
        'potassium': st.slider("⚡ Potassium (kg/ha)", 0, 400, 200)
    }
    st.header("🌤️ Weather Parameters")
    weather_params = {
        'rainfall': st.slider("☔ Annual Rainfall (mm)", 200, 3000, 1000),
        'temperature': st.slider("🌞 Average Temp (°C)", 10, 45, 25)
    }
    generate = st.button("🚀 Generate Crop Plan", key="generate")

if generate:
    if any(x is None for x in [crop_model, yield_model, crop_encoder, crop_encoder_yield, crop_prices, crop_season_mapping, reference_data]):
        st.error("⚠️ Models or data not loaded correctly.")
    else:
        with st.spinner("📊 Generating optimized crop cycle plan..."):
            plan = {}
            for season in ['Kharif', 'Rabi', 'Zaid']:
                season_crops = crop_season_mapping.get(season, [])
                best_crop = None
                best_profit = -float('inf')
                best_yield = 0
                for crop in season_crops:
                    if hasattr(crop_encoder_yield, "classes_") and crop in crop_encoder_yield.classes_:
                        crop_encoded = crop_encoder_yield.transform([crop])[0]
                        features = np.array([[crop_encoded, soil_params['ph'], soil_params['nitrogen'], weather_params['rainfall']]])
                        yield_pred = yield_model.predict(features)[0]
                        price_per_quintal = crop_prices.get(crop, 0)
                        total_yield_quintals = yield_pred * area_hectares * 10
                        gross_revenue = total_yield_quintals * price_per_quintal
                        cost_per_hectare = {
                            'Rice': 45000, 'Wheat': 40000, 'Cotton': 50000, 'Maize': 35000,
                            'Soybean': 30000, 'Sugarcane': 80000, 'Barley': 32000, 'Mustard': 28000
                        }
                        total_costs = cost_per_hectare.get(crop, 40000) * area_hectares
                        profit = gross_revenue - total_costs
                        if profit > best_profit:
                            best_profit = profit
                            best_crop = crop
                            best_yield = yield_pred
                if best_crop:
                    plan[season] = {
                        'crop': best_crop,
                        'yield_per_hectare': round(best_yield, 2),
                        'total_yield': round(best_yield * area_hectares, 2),
                        'profit': round(best_profit, 2),
                    }

            st.success("✅ Crop Cycle Plan Generated!")
            st.header("📋 Recommended Crop Cycle Plan")
            for season, data in plan.items():
                st.markdown(f"""
                <div class="metric-container">
                    <h3>{data['crop']}</h3>
                    <p><b>Yield per Hectare:</b> {data['yield_per_hectare']} tonnes</p>
                    <p><b>Total Yield:</b> {data['total_yield']} tonnes</p>
                    <p><b>Expected Profit:</b> ₹{data['profit']:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)

            if plan:
                seasons = list(plan.keys())
                profits = [plan[season]['profit'] for season in seasons]
                fig = px.bar(
                    x=seasons,
                    y=profits,
                    title="📊 Expected Profit by Season",
                    labels={'x': 'Season', 'y': 'Profit (₹)'},
                    color=profits,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #555; padding: 2rem;">
    <p>🌾 AI-Based Crop Cycle Planner | Developed for Agricultural Sustainability</p>
    <p><small>Government-grade system designed for accuracy, reliability, and user-friendliness</small></p>
</div>
""", unsafe_allow_html=True)
