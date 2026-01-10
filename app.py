import streamlit as st
import pandas as pd
import numpy as np
import json
from datetime import datetime

# --- CONFIG & STYLING ---
st.set_page_config(page_title="CropSense AI", page_icon="🌱", layout="wide")

def apply_custom_css():
    st.markdown("""
    <style>
        /* Fix for visibility: Force dark text on light backgrounds */
        .recommendation-card { 
            border-radius: 15px; 
            padding: 25px; 
            margin: 15px 0;
            border-left: 10px solid #2e7d32; 
            background-color: #ffffff !important; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            color: #1a1a1a !important;
        }
        .recommendation-card h2, .recommendation-card h3, .recommendation-card p, .recommendation-card span {
            color: #1a1a1a !important;
            margin-bottom: 5px;
        }
        .season-badge {
            background-color: #e8f5e9;
            color: #2e7d32 !important;
            padding: 4px 12px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9rem;
        }
        .metric-label { font-size: 1rem; color: #666; }
        .metric-value { font-size: 1.5rem; font-weight: bold; color: #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# --- DATA & LOGIC ---
@st.cache_data
def get_agricultural_data():
    # Crop price per quintal (100kg), Cost per hectare, Base yield in Tonnes/Hectare
    crop_info = {
        'Rice': {'price': 2500, 'cost': 45000, 'base_yield': 6.5},
        'Wheat': {'price': 2200, 'cost': 40000, 'base_yield': 4.5},
        'Cotton': {'price': 6000, 'cost': 50000, 'base_yield': 2.8},
        'Maize': {'price': 1800, 'cost': 35000, 'base_yield': 7.2},
        'Soybean': {'price': 4200, 'cost': 30000, 'base_yield': 3.5},
        'Sugarcane': {'price': 350, 'cost': 80000, 'base_yield': 65.0},
        'Barley': {'price': 1900, 'cost': 32000, 'base_yield': 4.0},
        'Mustard': {'price': 5500, 'cost': 28000, 'base_yield': 2.2}
    }
    seasons = {
        'Kharif (Monsoon)': ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
        'Rabi (Winter)': ['Wheat', 'Barley', 'Mustard', 'Maize'],
        'Zaid (Summer)': ['Rice', 'Maize']
    }
    return crop_info, seasons

crop_info, crop_seasons = get_agricultural_data()

def calculate_yield(crop, ph, rainfall, temp):
    base = crop_info[crop]['base_yield']
    # Simplified environmental impact logic
    ph_impact = 1.0 if 6.0 <= ph <= 7.5 else 0.8
    weather_impact = 1.0 if 20 <= temp <= 32 else 0.85
    rain_impact = 1.1 if 800 <= rainfall <= 1500 else 0.9
    return round(base * ph_impact * weather_impact * rain_impact, 2)

# --- SIDEBAR ---
with st.sidebar:
    st.title("🚜 Farm Manager")
    district = st.selectbox("Select District", ['Pune', 'Nashik', 'Nagpur', 'Satara', 'Aurangabad', 'Jalgaon', 'Kolhapur'])
    area = st.number_input("Farm Area (Hectares)", 0.5, 100.0, 2.0)
    
    st.divider()
    st.subheader("🧪 Soil Conditions")
    soil_ph = st.slider("Soil pH Level", 4.0, 10.0, 6.8)
    nitrogen = st.slider("Nitrogen Level (kg/ha)", 0, 500, 150)
    
    st.subheader("🌤️ Weather Forecast")
    rainfall = st.slider("Expected Annual Rain (mm)", 200, 3000, 1000)
    temperature = st.slider("Avg Temperature (°C)", 10, 45, 27)
    
    run_analysis = st.button("Generate Optimization Plan", use_container_width=True)

# --- MAIN PAGE ---
st.title("🌾 Maharashtra Crop Cycle Planner")
st.markdown(f"**Analysis for:** {district} District | **Land Size:** {area} Hectares")

if run_analysis:
    seasonal_recommendations = []
    
    for season, crops in crop_seasons.items():
        potential_crops = []
        for crop in crops:
            predicted_yield = calculate_yield(crop, soil_ph, rainfall, temperature)
            total_yield_tonnes = predicted_yield * area
            # Revenue = Total Tonnes * 10 (to get quintals) * Price per quintal
            gross_revenue = total_yield_tonnes * 10 * crop_info[crop]['price']
            total_cost = crop_info[crop]['cost'] * area
            profit = gross_revenue - total_cost
            
            potential_crops.append({
                'Season': season,
                'Crop': crop,
                'Yield_Per_Ha': predicted_yield,
                'Total_Yield': total_yield_tonnes,
                'Profit': profit
            })
        
        # Select best crop for the season
        best_crop = max(potential_crops, key=lambda x: x['Profit'])
        seasonal_recommendations.append(best_crop)

    # Summary Metrics
    df_results = pd.DataFrame(seasonal_recommendations)
    m1, m2, m3 = st.columns(3)
    total_annual_profit = df_results['Profit'].sum()
    
    m1.metric("Total Annual Profit", f"₹{total_annual_profit:,.0f}")
    m2.metric("Best Season", df_results.loc[df_results['Profit'].idxmax()]['Season'].split(' ')[0])
    m3.metric("Projected Yield (Annual)", f"{df_results['Total_Yield'].sum():.1f} Tons")

    # Chart Section
    st.subheader("📈 Seasonal Profit Analysis")
    # Using Streamlit Native Bar Chart (No Plotly Required)
    chart_df = df_results.set_index('Season')[['Profit']]
    st.bar_chart(chart_df, color="#2e7d32")

    # Detailed Cards
    st.subheader("📋 Recommended Crop Rotation")
    for index, row in df_results.iterrows():
        st.markdown(f"""
        <div class="recommendation-card">
            <span class="season-badge">{row['Season']}</span>
            <h2 style="margin-top:10px;">{row['Crop']}</h2>
            <div style="display: flex; gap: 40px; margin-top: 15px;">
                <div>
                    <p class="metric-label">Yield per Hectare</p>
                    <p class="metric-value">{row['Yield_Per_Ha']} Tons</p>
                </div>
                <div>
                    <p class="metric-label">Total Harvest</p>
                    <p class="metric-value">{row['Total_Yield']:.2f} Tons</p>
                </div>
                <div>
                    <p class="metric-label">Projected Profit</p>
                    <p class="metric-value">₹{row['Profit']:,.0f}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Expert Insight
    st.info(f"💡 **Expert Insight:** For {district}, rotating **{df_results.iloc[0]['Crop']}** with **{df_results.iloc[1]['Crop']}** is ideal for maintaining soil Nitrogen levels.")

else:
    st.info("👈 Fill in your farm details in the sidebar and click 'Generate Optimization Plan' to view your results.")
    # Placeholder Image for visual appeal
    st.image("https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&q=80&w=1200", caption="Smart Farming for a better yield")
