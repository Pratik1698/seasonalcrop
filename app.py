import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime

st.set_page_config(
    page_title="🌱 AI Crop Cycle Planner",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #2a5298;
    }
    .stButton > button {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🌾 AI-Based Crop Cycle Planner</h1>
    <p>Smart Agricultural Decision Support System</p>
</div>
""", unsafe_allow_html=True)

@st.cache_data
def get_sample_data():
    crop_data = {
        'crops': ['Rice', 'Wheat', 'Cotton', 'Maize', 'Soybean', 'Sugarcane', 'Barley', 'Mustard'],
        'states': ['Maharashtra', 'Punjab', 'Haryana', 'Uttar Pradesh', 'Gujarat', 'Rajasthan', 'Karnataka', 'Andhra Pradesh'],
        'districts': {
            'Maharashtra': ['Mumbai', 'Pune', 'Nagpur', 'Nashik'],
            'Punjab': ['Ludhiana', 'Amritsar', 'Jalandhar', 'Patiala'],
            'Haryana': ['Gurgaon', 'Faridabad', 'Panipat', 'Ambala'],
            'Uttar Pradesh': ['Lucknow', 'Kanpur', 'Agra', 'Varanasi'],
            'Gujarat': ['Ahmedabad', 'Surat', 'Vadodara', 'Rajkot'],
            'Rajasthan': ['Jaipur', 'Jodhpur', 'Udaipur', 'Kota'],
            'Karnataka': ['Bangalore', 'Mysore', 'Mangalore', 'Hubli'],
            'Andhra Pradesh': ['Hyderabad', 'Visakhapatnam', 'Vijayawada', 'Guntur']
        }
    }
    crop_prices = {
        'Rice': 2500, 'Wheat': 2200, 'Cotton': 6000, 'Maize': 1800,
        'Soybean': 4200, 'Sugarcane': 350, 'Barley': 1900, 'Mustard': 5500
    }
    crop_seasons = {
        'Kharif': ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
        'Rabi': ['Wheat', 'Barley', 'Mustard', 'Maize'],
        'Zaid': ['Rice', 'Maize', 'Cotton']
    }
    crop_costs = {
        'Rice': 45000, 'Wheat': 40000, 'Cotton': 50000, 'Maize': 35000,
        'Soybean': 30000, 'Sugarcane': 80000, 'Barley': 32000, 'Mustard': 28000
    }
    return crop_data, crop_prices, crop_seasons, crop_costs

crop_data, crop_prices, crop_seasons, crop_costs = get_sample_data()

def predict_crop_yield(crop, soil_ph, nitrogen, rainfall, temperature):
    base_yields = {
        'Rice': 6.5, 'Wheat': 4.5, 'Cotton': 2.8, 'Maize': 7.2,
        'Soybean': 3.5, 'Sugarcane': 65.0, 'Barley': 4.0, 'Mustard': 2.2
    }
    base_yield = base_yields.get(crop, 4.0)
    ph_factor = 1.0 if 6.0 <= soil_ph <= 7.5 else 0.8
    nitrogen_factor = min(1.2, nitrogen / 200)
    rainfall_factor = min(1.1, rainfall / 800) if rainfall < 1200 else max(0.9, 1200 / rainfall)
    temp_factor = 1.0 if 20 <= temperature <= 30 else 0.85
    return round(base_yield * ph_factor * nitrogen_factor * rainfall_factor * temp_factor, 2)

def calculate_profit(crop, yield_per_ha, area_hectares, crop_prices, crop_costs):
    total_yield = yield_per_ha * area_hectares
    gross_revenue = total_yield * 10 * crop_prices.get(crop, 2000)
    total_costs = crop_costs.get(crop, 40000) * area_hectares
    return round(gross_revenue - total_costs, 2)

with st.sidebar:
    st.header("📋 Farm Information")
    state = st.selectbox("📍 Select State", crop_data['states'])
    district = st.selectbox("📍 Select District", crop_data['districts'][state])
    area_hectares = st.number_input("🏞️ Farm Area (hectares)", 0.1, 1000.0, 2.0, step=0.1)

    st.header("🌱 Soil Parameters")
    soil_ph = st.slider("🌡️ Soil pH", 4.0, 10.0, 7.0, step=0.1)
    nitrogen = st.slider("🌿 Nitrogen (kg/ha)", 0, 500, 250)
    phosphorus = st.slider("💧 Phosphorus (kg/ha)", 0, 100, 50)
    potassium = st.slider("⚡ Potassium (kg/ha)", 0, 400, 200)

    st.header("🌤️ Weather Parameters")
    rainfall = st.slider("☔ Annual Rainfall (mm)", 200, 3000, 1000)
    temperature = st.slider("🌞 Average Temperature (°C)", 10, 45, 25)

    generate_plan = st.button("🚀 Generate Crop Plan")

if generate_plan:
    with st.spinner("📊 Analyzing your farm conditions..."):
        recommendations = {}
        for season, season_crops in crop_seasons.items():
            best_crop, best_profit, best_yield = None, -float('inf'), 0
            for crop in season_crops:
                predicted_yield = predict_crop_yield(crop, soil_ph, nitrogen, rainfall, temperature)
                profit = calculate_profit(crop, predicted_yield, area_hectares, crop_prices, crop_costs)
                if profit > best_profit:
                    best_profit, best_crop, best_yield = profit, crop, predicted_yield
            recommendations[season] = {
                'crop': best_crop,
                'yield_per_hectare': best_yield,
                'total_yield': round(best_yield * area_hectares, 2),
                'profit': best_profit,
                'price_per_quintal': crop_prices.get(best_crop, 2000),
                'cost_per_hectare': crop_costs.get(best_crop, 40000)
            }

        st.success("🎉 Crop Cycle Plan Generated Successfully!")
        st.header("📋 Recommended Crop Cycle Plan")

        cols = st.columns(3)
        for i, (season, data) in enumerate(recommendations.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🌱 {season} Season</h3>
                    <h4 style="color: #2a5298;">{data['crop']}</h4>
                    <p><b>Yield per Hectare:</b> {data['yield_per_hectare']} tonnes</p>
                    <p><b>Total Yield:</b> {data['total_yield']} tonnes</p>
                    <p><b>Expected Profit:</b> ₹{data['profit']:,.0f}</p>
                    <p><b>Price per Quintal:</b> ₹{data['price_per_quintal']}</p>
                </div>
                """, unsafe_allow_html=True)

        # ✅ Matplotlib chart instead of Plotly
        st.header("📊 Profit Comparison by Season")
        chart_data = pd.DataFrame({
            'Season': list(recommendations.keys()),
            'Profit': [recommendations[s]['profit'] for s in recommendations],
            'Crop': [recommendations[s]['crop'] for s in recommendations]
        })
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(chart_data['Season'], chart_data['Profit'])
        ax.set_ylabel("Profit (₹)")
        ax.set_title("Expected Profit by Season")

        # Add crop labels on top of bars
        for bar, crop in zip(bars, chart_data['Crop']):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    crop,
                    ha='center', va='bottom')
        st.pyplot(fig)

        st.header("💡 Key Insights")
        total_annual_profit = sum(r['profit'] for r in recommendations.values())
        best_season = max(recommendations, key=lambda x: recommendations[x]['profit'])
        total_yield = sum(r['total_yield'] for r in recommendations.values())

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Annual Profit", f"₹{total_annual_profit:,.0f}",
                    delta=f"Per hectare: ₹{total_annual_profit/area_hectares:,.0f}")
        col2.metric("Best Season", best_season, delta=f"{recommendations[best_season]['crop']}")
        col3.metric("Total Annual Yield", f"{total_yield:.1f} tonnes",
                    delta=f"Per hectare: {total_yield/area_hectares:.1f} tonnes")

        st.header("💾 Save Results")
        if st.button("📥 Download Report as JSON"):
            report_data = {
                'farm_info': {'state': state,'district': district,'area_hectares': area_hectares},
                'soil_parameters': {'ph': soil_ph,'nitrogen': nitrogen,'phosphorus': phosphorus,'potassium': potassium},
                'weather_parameters': {'rainfall': rainfall,'temperature': temperature},
                'recommendations': recommendations,
                'total_annual_profit': total_annual_profit,
                'generated_on': datetime.now().isoformat()
            }
            st.download_button("📋 Download Complete Report",
                               data=json.dumps(report_data, indent=2),
                               file_name=f"crop_plan_{state}_{datetime.now().strftime('%Y%m%d')}.json",
                               mime="application/json")

else:
    st.info("👈 Please fill in your farm details in the sidebar and click 'Generate Crop Plan' to get started!")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌾 AI-Based Crop Cycle Planner | Developed for Agricultural Sustainability</p>
    <p><small>Supporting farmers with data-driven crop selection decisions</small></p>
</div>
""", unsafe_allow_html=True)