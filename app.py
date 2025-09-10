import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import os
from datetime import datetime

# CRITICAL: This MUST be the very first Streamlit command
st.set_page_config(
    page_title="🌱 AI Crop Cycle Planner",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Show loading progress
with st.spinner("🚀 Loading Crop Cycle Planner..."):
    st.write("✅ Page config loaded")

# Simplified CSS
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

st.write("✅ CSS loaded")

# Header
st.markdown("""
<div class="main-header">
    <h1>🌾 AI-Based Crop Cycle Planner</h1>
    <p>Smart Agricultural Decision Support System</p>
</div>
""", unsafe_allow_html=True)

st.write("✅ Header loaded")

# Create mock data if files don't exist
@st.cache_data
def get_sample_data():
    """Create sample data for the application"""
    
    # Sample crop data
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
    
    # Crop prices (per quintal)
    crop_prices = {
        'Rice': 2500, 'Wheat': 2200, 'Cotton': 6000, 'Maize': 1800,
        'Soybean': 4200, 'Sugarcane': 350, 'Barley': 1900, 'Mustard': 5500
    }
    
    # Seasonal mapping
    crop_seasons = {
        'Kharif': ['Rice', 'Cotton', 'Maize', 'Soybean', 'Sugarcane'],
        'Rabi': ['Wheat', 'Barley', 'Mustard', 'Maize'],
        'Zaid': ['Rice', 'Maize', 'Cotton']
    }
    
    # Cost per hectare
    crop_costs = {
        'Rice': 45000, 'Wheat': 40000, 'Cotton': 50000, 'Maize': 35000,
        'Soybean': 30000, 'Sugarcane': 80000, 'Barley': 32000, 'Mustard': 28000
    }
    
    return crop_data, crop_prices, crop_seasons, crop_costs

# Load sample data
try:
    crop_data, crop_prices, crop_seasons, crop_costs = get_sample_data()
    st.write("✅ Sample data loaded")
except Exception as e:
    st.error(f"❌ Error loading sample data: {e}")
    st.stop()

# Prediction functions (simplified)
def predict_crop_yield(crop, soil_ph, nitrogen, rainfall, temperature):
    """Simplified yield prediction based on environmental factors"""
    
    # Base yields (tonnes per hectare)
    base_yields = {
        'Rice': 6.5, 'Wheat': 4.5, 'Cotton': 2.8, 'Maize': 7.2,
        'Soybean': 3.5, 'Sugarcane': 65.0, 'Barley': 4.0, 'Mustard': 2.2
    }
    
    base_yield = base_yields.get(crop, 4.0)
    
    # Adjust based on conditions
    ph_factor = 1.0 if 6.0 <= soil_ph <= 7.5 else 0.8
    nitrogen_factor = min(1.2, nitrogen / 200)
    rainfall_factor = min(1.1, rainfall / 800) if rainfall < 1200 else max(0.9, 1200 / rainfall)
    temp_factor = 1.0 if 20 <= temperature <= 30 else 0.85
    
    final_yield = base_yield * ph_factor * nitrogen_factor * rainfall_factor * temp_factor
    return round(final_yield, 2)

def calculate_profit(crop, yield_per_ha, area_hectares, crop_prices, crop_costs):
    """Calculate profit for given crop"""
    total_yield = yield_per_ha * area_hectares
    total_yield_quintals = total_yield * 10  # Convert tonnes to quintals
    gross_revenue = total_yield_quintals * crop_prices.get(crop, 2000)
    total_costs = crop_costs.get(crop, 40000) * area_hectares
    profit = gross_revenue - total_costs
    return round(profit, 2)

st.write("✅ Prediction functions loaded")

# Sidebar inputs
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

st.write("✅ Sidebar loaded")

# Main content
if generate_plan:
    st.write("✅ Generate button clicked")
    
    with st.spinner("📊 Analyzing your farm conditions..."):
        try:
            # Generate crop recommendations for each season
            recommendations = {}
            
            for season, season_crops in crop_seasons.items():
                st.write(f"🔍 Analyzing {season} season...")
                
                best_crop = None
                best_profit = -float('inf')
                best_yield = 0
                
                for crop in season_crops:
                    # Predict yield
                    predicted_yield = predict_crop_yield(crop, soil_ph, nitrogen, rainfall, temperature)
                    
                    # Calculate profit
                    profit = calculate_profit(crop, predicted_yield, area_hectares, crop_prices, crop_costs)
                    
                    if profit > best_profit:
                        best_profit = profit
                        best_crop = crop
                        best_yield = predicted_yield
                
                recommendations[season] = {
                    'crop': best_crop,
                    'yield_per_hectare': best_yield,
                    'total_yield': round(best_yield * area_hectares, 2),
                    'profit': best_profit,
                    'price_per_quintal': crop_prices.get(best_crop, 2000),
                    'cost_per_hectare': crop_costs.get(best_crop, 40000)
                }
            
            st.write("✅ Analysis complete")
            
            # Display results
            st.success("🎉 Crop Cycle Plan Generated Successfully!")
            
            st.header("📋 Recommended Crop Cycle Plan")
            
            # Create columns for better layout
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
            
            # Profit comparison chart
            st.header("📊 Profit Comparison by Season")
            
            seasons = list(recommendations.keys())
            profits = [recommendations[season]['profit'] for season in seasons]
            crops = [recommendations[season]['crop'] for season in seasons]
            
            # Create DataFrame for the chart
            chart_data = pd.DataFrame({
                'Season': seasons,
                'Profit (₹)': profits,
                'Crop': crops
            })
            
            # Create bar chart
            fig = px.bar(
                chart_data,
                x='Season',
                y='Profit (₹)',
                color='Profit (₹)',
                text='Crop',
                title="Expected Profit by Season",
                color_continuous_scale='Blues'
            )
            
            fig.update_traces(textposition='outside')
            fig.update_layout(height=500)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights
            st.header("💡 Key Insights")
            
            total_annual_profit = sum(profits)
            best_season = max(recommendations.keys(), key=lambda x: recommendations[x]['profit'])
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Total Annual Profit",
                    value=f"₹{total_annual_profit:,.0f}",
                    delta=f"Per hectare: ₹{total_annual_profit/area_hectares:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Best Season",
                    value=best_season,
                    delta=f"{recommendations[best_season]['crop']}"
                )
            
            with col3:
                total_yield = sum([recommendations[season]['total_yield'] for season in seasons])
                st.metric(
                    label="Total Annual Yield",
                    value=f"{total_yield:.1f} tonnes",
                    delta=f"Per hectare: {total_yield/area_hectares:.1f} tonnes"
                )
            
            # Save results option
            st.header("💾 Save Results")
            
            if st.button("📥 Download Report as JSON"):
                report_data = {
                    'farm_info': {
                        'state': state,
                        'district': district,
                        'area_hectares': area_hectares
                    },
                    'soil_parameters': {
                        'ph': soil_ph,
                        'nitrogen': nitrogen,
                        'phosphorus': phosphorus,
                        'potassium': potassium
                    },
                    'weather_parameters': {
                        'rainfall': rainfall,
                        'temperature': temperature
                    },
                    'recommendations': recommendations,
                    'total_annual_profit': total_annual_profit,
                    'generated_on': datetime.now().isoformat()
                }
                
                st.download_button(
                    label="📋 Download Complete Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"crop_plan_{state}_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )
        
        except Exception as e:
            st.error(f"❌ Error generating recommendations: {e}")
            st.error("Please check your inputs and try again.")

else:
    # Show instructions when app first loads
    st.info("👈 Please fill in your farm details in the sidebar and click 'Generate Crop Plan' to get started!")
    
    # Show sample data info
    with st.expander("📊 About this Application"):
        st.write("""
        This AI-Based Crop Cycle Planner helps farmers make informed decisions about crop selection
        based on:
        
        - **Soil Conditions**: pH, Nitrogen, Phosphorus, Potassium levels
        - **Weather Patterns**: Rainfall and temperature data
        - **Economic Factors**: Market prices and cultivation costs
        - **Seasonal Optimization**: Recommendations for Kharif, Rabi, and Zaid seasons
        
        The app uses simplified prediction models to estimate yields and calculate expected profits
        for different crops in each season.
        """)
        
        st.write("**Supported Crops:**")
        for season, crops in crop_seasons.items():
            st.write(f"- **{season}:** {', '.join(crops)}")

st.write("✅ App fully loaded")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌾 AI-Based Crop Cycle Planner | Developed for Agricultural Sustainability</p>
    <p><small>Supporting farmers with data-driven crop selection decisions</small></p>
</div>
""", unsafe_allow_html=True)

st.write("✅ Footer loaded")