
import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="🌾 AI Crop Cycle Planner Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #2E8B57, #228B22);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #228B22;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
    .info-box {
        background: #f0f8ff;
        border-left: 4px solid #1e90ff;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 8px;
    }
    .stButton > button {
        background: linear-gradient(90deg, #228B22, #32CD32);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: bold;
        width: 100%;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# Data loading and preprocessing functions
@st.cache_data
def load_and_prepare_data():
    """Load and prepare all datasets for ML models"""
    try:
        # Load the uploaded files
        crop_yield_df = pd.read_csv('crop_yield_data.csv')
        soil_df = pd.read_csv('soil_data.csv')
        weather_df = pd.read_csv('weather_data.csv')
        market_df = pd.read_csv('market_prices.csv')

        # Load JSON files
        with open('crop_season_mapping.json', 'r') as f:
            season_mapping = json.load(f)

        with open('crop_prices.json', 'r') as f:
            crop_prices = json.load(f)

        with open('reference_data.json', 'r') as f:
            reference_data = json.load(f)

        return {
            'crop_yield': crop_yield_df,
            'soil': soil_df,
            'weather': weather_df,
            'market': market_df,
            'season_mapping': season_mapping,
            'crop_prices': crop_prices,
            'reference': reference_data
        }
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def create_ml_dataset(data_dict):
    """Create a comprehensive dataset for ML model training"""

    crop_yield_df = data_dict['crop_yield']
    soil_df = data_dict['soil']
    weather_df = data_dict['weather']
    market_df = data_dict['market']

    # Aggregate weather data by year and location
    weather_agg = weather_df.groupby(['State', 'District', 'Year']).agg({
        'Rainfall_mm': 'sum',
        'Temperature_C': 'mean',
        'Humidity': 'mean'
    }).reset_index()

    # Get average soil parameters by location
    soil_agg = soil_df.groupby(['State', 'District']).agg({
        'pH': 'mean',
        'Nitrogen': 'mean',
        'Phosphorus': 'mean',
        'Potassium': 'mean',
        'Organic_Carbon': 'mean',
        'Moisture': 'mean'
    }).reset_index()

    # Get average market prices by year
    market_agg = market_df.groupby(['State', 'Year', 'Crop']).agg({
        'Price_per_Quintal': 'mean'
    }).reset_index()

    # Merge datasets
    ml_data = crop_yield_df.merge(weather_agg, on=['State', 'District', 'Year'], how='left')
    ml_data = ml_data.merge(soil_agg, on=['State', 'District'], how='left')
    ml_data = ml_data.merge(market_agg, on=['State', 'Year', 'Crop'], how='left')

    # Fill missing values
    ml_data = ml_data.fillna(ml_data.mean(numeric_only=True))

    return ml_data

class CropRecommendationSystem:
    def __init__(self):
        self.yield_model = None
        self.crop_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None

    def prepare_features(self, ml_data):
        """Prepare features for ML models"""

        # Encode categorical variables
        state_encoder = LabelEncoder()
        district_encoder = LabelEncoder()
        season_encoder = LabelEncoder()

        ml_data['State_encoded'] = state_encoder.fit_transform(ml_data['State'])
        ml_data['District_encoded'] = district_encoder.fit_transform(ml_data['District'])
        ml_data['Season_encoded'] = season_encoder.fit_transform(ml_data['Season'])

        # Feature columns for prediction
        feature_columns = [
            'State_encoded', 'District_encoded', 'Season_encoded', 'Year',
            'Area_Hectares', 'Rainfall_mm', 'Temperature_C', 'Humidity',
            'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Moisture'
        ]

        self.feature_columns = feature_columns

        # Store encoders for later use
        self.state_encoder = state_encoder
        self.district_encoder = district_encoder 
        self.season_encoder = season_encoder

        return ml_data[feature_columns], ml_data['Yield_per_Hectare'], ml_data['Crop']

    def train_models(self, ml_data):
        """Train yield prediction and crop recommendation models"""

        X, y_yield, y_crop = self.prepare_features(ml_data)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode crop labels
        y_crop_encoded = self.label_encoder.fit_transform(y_crop)

        # Split data
        X_train, X_test, y_yield_train, y_yield_test = train_test_split(
            X_scaled, y_yield, test_size=0.2, random_state=42
        )

        X_train_crop, X_test_crop, y_crop_train, y_crop_test = train_test_split(
            X_scaled, y_crop_encoded, test_size=0.2, random_state=42
        )

        # Train yield prediction model (Regression)
        self.yield_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.yield_model.fit(X_train, y_yield_train)

        # Train crop recommendation model (Classification)
        self.crop_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            random_state=42
        )
        self.crop_model.fit(X_train_crop, y_crop_train)

        # Calculate performance metrics
        yield_pred = self.yield_model.predict(X_test)
        crop_pred = self.crop_model.predict(X_test_crop)

        yield_r2 = r2_score(y_yield_test, yield_pred)
        yield_rmse = np.sqrt(mean_squared_error(y_yield_test, yield_pred))
        crop_accuracy = accuracy_score(y_crop_test, crop_pred)

        return {
            'yield_r2': yield_r2,
            'yield_rmse': yield_rmse,
            'crop_accuracy': crop_accuracy
        }

    def predict_yield(self, input_features):
        """Predict crop yield based on input features"""
        if self.yield_model is None:
            return None

        features_scaled = self.scaler.transform([input_features])
        prediction = self.yield_model.predict(features_scaled)[0]
        return max(0, prediction)  # Ensure non-negative yield

    def recommend_crop(self, input_features):
        """Recommend best crop based on input features"""
        if self.crop_model is None:
            return None, None

        features_scaled = self.scaler.transform([input_features])

        # Get probabilities for all crops
        probabilities = self.crop_model.predict_proba(features_scaled)[0]
        crop_names = self.label_encoder.classes_

        # Get top 3 recommendations
        top_indices = np.argsort(probabilities)[-3:][::-1]
        recommendations = []

        for idx in top_indices:
            recommendations.append({
                'crop': crop_names[idx],
                'probability': probabilities[idx]
            })

        return recommendations

    def calculate_profit(self, crop, yield_per_ha, area_hectares, data_dict):
        """Calculate expected profit for a crop"""

        # Get crop price
        crop_prices = data_dict['crop_prices']
        price_per_quintal = crop_prices.get(crop, 2000)  # Default price

        # Calculate revenue (yield in tonnes * 10 quintals/tonne * price)
        total_yield_tonnes = yield_per_ha * area_hectares
        total_yield_quintals = total_yield_tonnes * 10
        gross_revenue = total_yield_quintals * price_per_quintal

        # Estimate costs (simplified)
        cost_per_hectare = {
            'Rice': 45000, 'Wheat': 40000, 'Cotton': 55000, 'Maize': 35000,
            'Soybean': 30000, 'Sugarcane': 80000, 'Barley': 32000, 'Mustard': 28000,
            'Gram': 25000, 'Peas': 30000, 'Fodder': 20000, 'Vegetables': 60000,
            'Watermelon': 40000, 'Cucumber': 45000
        }

        total_cost = cost_per_hectare.get(crop, 40000) * area_hectares
        profit = gross_revenue - total_cost

        return {
            'revenue': gross_revenue,
            'cost': total_cost,
            'profit': profit,
            'roi': (profit / total_cost * 100) if total_cost > 0 else 0
        }

# Initialize the app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AI-Powered Crop Cycle Planner Pro</h1>
        <p>Advanced Machine Learning for Smart Agricultural Decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("🔄 Loading agricultural datasets..."):
        data_dict = load_and_prepare_data()

    if data_dict is None:
        st.error("❌ Failed to load data files. Please ensure all data files are available.")
        return

    # Create ML dataset
    with st.spinner("🧠 Preparing machine learning models..."):
        ml_data = create_ml_dataset(data_dict)

        # Initialize and train models
        crop_system = CropRecommendationSystem()
        model_performance = crop_system.train_models(ml_data)

    st.success("✅ AI models trained successfully!")

    # Display model performance
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Yield Model Accuracy</h4>
            <h2>{model_performance['yield_r2']:.3f}</h2>
            <p>R² Score</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h4>📈 Prediction Error</h4>
            <h2>{model_performance['yield_rmse']:.2f}</h2>
            <p>RMSE (tonnes/ha)</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h4>🎯 Crop Model Accuracy</h4>
            <h2>{model_performance['crop_accuracy']:.3f}</h2>
            <p>Classification Score</p>
        </div>
        """, unsafe_allow_html=True)

    # Sidebar for input parameters
    st.sidebar.header("📋 Farm & Environmental Parameters")

    # Location inputs
    st.sidebar.subheader("📍 Location")
    selected_state = st.sidebar.selectbox("State", data_dict['reference']['states'])
    available_districts = data_dict['reference']['districts'].get(selected_state, ['Other'])
    selected_district = st.sidebar.selectbox("District", available_districts)

    # Farm parameters
    st.sidebar.subheader("🏞️ Farm Information")
    area_hectares = st.sidebar.number_input("Farm Area (hectares)", 0.1, 1000.0, 2.0, step=0.1)
    season = st.sidebar.selectbox("Season", data_dict['reference']['seasons'])
    year = st.sidebar.number_input("Year", 2015, 2030, 2024)

    # Soil parameters
    st.sidebar.subheader("🌱 Soil Parameters")
    ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 7.0, step=0.1)
    nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 150, 400, 275)
    phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 20, 80, 50)
    potassium = st.sidebar.slider("Potassium (kg/ha)", 150, 350, 250)
    organic_carbon = st.sidebar.slider("Organic Carbon (%)", 0.3, 1.5, 0.9, step=0.1)
    moisture = st.sidebar.slider("Soil Moisture (%)", 15, 45, 30)

    # Weather parameters
    st.sidebar.subheader("🌤️ Weather Conditions")
    rainfall = st.sidebar.slider("Annual Rainfall (mm)", 300, 2000, 1000)
    temperature = st.sidebar.slider("Average Temperature (°C)", 15, 40, 27)
    humidity = st.sidebar.slider("Humidity (%)", 40, 85, 63)

    # Prediction button
    if st.sidebar.button("🚀 Generate AI Recommendations"):

        # Prepare input features
        try:
            # Encode inputs
            state_encoded = crop_system.state_encoder.transform([selected_state])[0]
            district_encoded = crop_system.district_encoder.transform([selected_district])[0] if selected_district in crop_system.district_encoder.classes_ else 0
            season_encoded = crop_system.season_encoder.transform([season])[0]

            input_features = [
                state_encoded, district_encoded, season_encoded, year, area_hectares,
                rainfall, temperature, humidity, ph, nitrogen, phosphorus, 
                potassium, organic_carbon, moisture
            ]

            # Get crop recommendations
            recommendations = crop_system.recommend_crop(input_features)

            st.header("🤖 AI-Generated Recommendations")

            # Display top 3 crop recommendations
            for i, rec in enumerate(recommendations):
                crop_name = rec['crop']
                probability = rec['probability']

                # Predict yield for this crop
                predicted_yield = crop_system.predict_yield(input_features)

                # Calculate profit
                profit_info = crop_system.calculate_profit(
                    crop_name, predicted_yield, area_hectares, data_dict
                )

                # Display recommendation card
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h3>#{i+1} Recommended Crop: {crop_name}</h3>
                        <div style="display: flex; justify-content: space-around; margin-top: 1rem;">
                            <div>
                                <h4>{predicted_yield:.2f}</h4>
                                <p>Tonnes/Hectare</p>
                            </div>
                            <div>
                                <h4>{probability*100:.1f}%</h4>
                                <p>Confidence</p>
                            </div>
                            <div>
                                <h4>₹{profit_info['profit']:,.0f}</h4>
                                <p>Expected Profit</p>
                            </div>
                            <div>
                                <h4>{profit_info['roi']:.1f}%</h4>
                                <p>ROI</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col2:
                    # Create a small chart for this recommendation
                    fig = go.Figure(data=go.Bar(
                        x=['Revenue', 'Cost', 'Profit'],
                        y=[profit_info['revenue'], profit_info['cost'], profit_info['profit']],
                        marker_color=['lightgreen', 'lightcoral', 'gold']
                    ))
                    fig.update_layout(
                        title=f"{crop_name} Economics",
                        height=300,
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # Seasonal analysis
            st.header("📅 Seasonal Analysis")

            season_crops = data_dict['season_mapping'].get(season, [])
            if season_crops:
                st.info(f"**{season} Season Crops:** {', '.join(season_crops)}")

                # Compare recommendations with seasonal crops
                recommended_crops = [rec['crop'] for rec in recommendations]
                seasonal_matches = set(recommended_crops) & set(season_crops)

                if seasonal_matches:
                    st.success(f"✅ AI recommendations match seasonal crops: {', '.join(seasonal_matches)}")
                else:
                    st.warning("⚠️ AI recommendations differ from traditional seasonal crops. Consider market opportunities!")

            # Environmental suitability analysis
            st.header("🌍 Environmental Suitability Analysis")

            # Create environmental factor visualization
            factors = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'Temperature']
            values = [ph, nitrogen/400*100, phosphorus, potassium/350*100, rainfall/2000*100, temperature/40*100]

            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=factors,
                fill='toself',
                name='Current Conditions'
            ))

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Environmental Conditions Profile"
            )

            st.plotly_chart(fig, use_container_width=True)

            # Risk assessment
            st.header("⚠️ Risk Assessment")

            risks = []
            if ph < 6.0 or ph > 8.0:
                risks.append("🔴 Soil pH outside optimal range (6.0-8.0)")
            if rainfall < 500:
                risks.append("🟡 Low rainfall may require irrigation")
            if temperature > 35:
                risks.append("🟠 High temperature stress possible")
            if nitrogen < 200:
                risks.append("🟡 Low nitrogen levels - fertilizer needed")

            if risks:
                for risk in risks:
                    st.warning(risk)
            else:
                st.success("✅ Environmental conditions are favorable!")

        except Exception as e:
            st.error(f"❌ Error generating recommendations: {str(e)}")
            st.error("Please check your inputs and try again.")

    # Data insights section
    with st.expander("📊 Agricultural Data Insights"):

        # Crop yield comparison
        st.subheader("🌾 Crop Yield Comparison")

        yield_data = ml_data.groupby('Crop')['Yield_per_Hectare'].mean().sort_values(ascending=False)

        fig = px.bar(
            x=yield_data.values,
            y=yield_data.index,
            orientation='h',
            title="Average Yield by Crop Type",
            labels={'x': 'Yield (tonnes/hectare)', 'y': 'Crop'}
        )
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)

        # State-wise analysis
        st.subheader("🗺️ Regional Analysis")

        state_yield = ml_data.groupby('State')['Yield_per_Hectare'].mean().sort_values(ascending=False)

        fig = px.bar(
            x=state_yield.index,
            y=state_yield.values,
            title="Average Yield by State",
            labels={'x': 'State', 'y': 'Average Yield (tonnes/hectare)'}
        )
        st.plotly_chart(fig, use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌾 AI-Powered Crop Cycle Planner Pro | Advanced Agricultural Intelligence</p>
        <p><small>Empowering farmers with data-driven decisions for sustainable agriculture</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
