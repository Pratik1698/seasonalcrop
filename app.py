import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px  # Upgraded visualization
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="CropSense AI", page_icon="🌱", layout="wide")

def local_css():
    st.markdown("""
    <style>
        .main { background-color: #f8f9fa; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .recommendation-card { 
            border-radius: 15px; padding: 20px; margin: 10px 0;
            border-top: 5px solid #2e7d32; background: white;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .sidebar-header { color: #2e7d32; font-weight: bold; margin-top: 1rem; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- DATA & LOGIC ---
@st.cache_data
def get_extended_data():
    # Adding N-P-K requirements for more scientific logic
    crop_info = {
        'Rice': {'price': 2500, 'cost': 45000, 'base_yield': 6.5, 'npk': [120, 60, 60]},
        'Wheat': {'price': 2200, 'cost': 40000, 'base_yield': 4.5, 'npk': [100, 50, 40]},
        'Cotton': {'price': 6000, 'cost': 50000, 'base_yield': 2.8, 'npk': [100, 50, 50]},
        'Maize': {'price': 1800, 'cost': 35000, 'base_yield': 7.2, 'npk': [120, 60, 40]},
        'Soybean': {'price': 4200, 'cost': 30000, 'base_yield': 3.5, 'npk': [20, 60, 40]},
        'Sugarcane': {'price': 350, 'cost': 80000, 'base_yield': 65.0, 'npk': [250, 115, 115]},
        'Mustard': {'price': 5500, 'cost': 28000, 'base_yield': 2.2, 'npk': [80, 40, 40]}
    }
    seasons = {
        'Kharif (Monsoon)': ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
        'Rabi (Winter)': ['Wheat', 'Maize', 'Mustard'],
        'Zaid (Summer)': ['Rice', 'Maize']
    }
    return crop_info, seasons

crop_info, crop_seasons = get_extended_data()

def predict_crop_yield(crop, ph, n, p, k, rain, temp):
    data = crop_info[crop]
    # Yield penalty logic based on NPK sufficiency
    n_ratio = min(1.0, n / data['npk'][0])
    p_ratio = min(1.0, p / data['npk'][1])
    k_ratio = min(1.0, k / data['npk'][2])
    nutrient_factor = (n_ratio + p_ratio + k_ratio) / 3
    
    ph_factor = 1.0 if 6.0 <= ph <= 7.5 else 0.75
    temp_factor = 1.0 if 18 <= temp <= 32 else 0.8
    
    return round(data['base_yield'] * nutrient_factor * ph_factor * temp_factor, 2)

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2942/2942257.png", width=100)
    st.title("Farm Dashboard")
    
    district = st.selectbox("📍 District (Maharashtra)", ['Pune', 'Nashik', 'Nagpur', 'Satara', 'Aurangabad'])
    area = st.number_input("🏞️ Farm Area (Hectares)", 0.5, 500.0, 1.0)
    
    st.markdown("### 🧪 Soil Analysis")
    ph = st.slider("Soil pH", 4.0, 10.0, 6.5)
    n = st.number_input("Nitrogen (kg/ha)", 0, 500, 150)
    p = st.number_input("Phosphorus (kg/ha)", 0, 200, 50)
    k = st.number_input("Potassium (kg/ha)", 0, 400, 100)
    
    st.markdown("### ☁️ Climate Forecast")
    rainfall = st.slider("Expected Rainfall (mm)", 200, 3000, 1100)
    temp = st.slider("Avg Temperature (°C)", 10, 45, 28)
    
    run_btn = st.button("Analyze & Optimize", use_container_width=True)

# --- MAIN CONTENT ---
if run_btn:
    results = []
    for season, crops in crop_seasons.items():
        season_results = []
        for crop in crops:
            y = predict_crop_yield(crop, ph, n, p, k, rainfall, temp)
            revenue = y * 10 * crop_info[crop]['price'] * area # Converting metric tons to quintals
            cost = crop_info[crop]['cost'] * area
            profit = revenue - cost
            season_results.append({'Crop': crop, 'Yield': y, 'Profit': profit, 'Season': season})
        
        # Pick the best for each season
        best = max(season_results, key=lambda x: x['Profit'])
        results.append(best)

    df_results = pd.DataFrame(results)

    # Metrics Row
    st.title(f"🌾 Strategy for {district} District")
    m1, m2, m3 = st.columns(3)
    m1.metric("Annual Profit Est.", f"₹{df_results['Profit'].sum():,.0f}")
    m2.metric("Peak Season", df_results.loc[df_results['Profit'].idxmax()]['Season'])
    m3.metric("Avg. Yield/Ha", f"{df_results['Yield'].mean():.2f} T")

    # Visualizations
    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.subheader("Seasonal Profitability Comparison")
        fig = px.bar(df_results, x='Season', y='Profit', color='Crop', 
                     text_auto='.2s', color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Nutrient Sufficiency")
        # Visualizing NPK vs Required for the best Kharif crop
        best_kharif = df_results.iloc[0]['Crop']
        req_npk = crop_info[best_kharif]['npk']
        npk_df = pd.DataFrame({
            'Nutrient': ['N', 'P', 'K'],
            'Actual': [n, p, k],
            'Required': req_npk
        })
        fig_npk = px.line_polar(npk_df, r='Actual', theta='Nutrient', line_close=True)
        st.plotly_chart(fig_npk, use_container_width=True)

    # Detailed Cards
    st.subheader("Recommended Action Plan")
    for _, row in df_results.iterrows():
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <div style="display: flex; justify-content: space-between;">
                    <span style="font-size: 1.2rem; font-weight: bold;">{row['Season']}</span>
                    <span style="color: #2e7d32; font-weight: bold;">Rank #1 Recommendation</span>
                </div>
                <h2 style="margin: 10px 0;">{row['Crop']}</h2>
                <p>Estimated Yield: <b>{row['Yield']} Tons/Ha</b> | Potential Profit: <b>₹{row['Profit']:,.2f}</b></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Fertilizer Insight
    st.info(f"💡 **Tip:** Your soil Nitrogen is {'low' if n < 100 else 'optimal'}. Consider adding urea if planting {df_results.iloc[0]['Crop']}.")

else:
    st.markdown("""
    ### Welcome to the Crop Cycle Planner!
    Please enter your soil test results and farm area in the sidebar to generate:
    * **Optimal crop rotations** for 3 seasons.
    * **Profitability forecasts** based on current market prices.
    * **Nutrient gap analysis** for your specific soil profile.
    """)
    st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&q=80&w=1000", use_container_width=True)
