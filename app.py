import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# Page config & CSS (unchanged)
# ------------------------------
st.set_page_config(
    page_title="🌾 AI Crop Cycle Planner Pro",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# ------------------------------
# Data loading / prepare
# ------------------------------
@st.cache_data
def load_json_file(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data
def load_csv_file(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

@st.cache_data
def load_and_prepare_data(file_paths=None):
    """
    Try to load canonical files; if missing, return None and allow user to upload via UI.
    """
    # default expected files
    paths = file_paths or {
        'crop_yield': 'crop_yield_data.csv',
        'soil': 'soil_data.csv',
        'weather': 'weather_data.csv',
        'market': 'market_prices.csv',
        'season_mapping': 'crop_season_mapping.json',
        'crop_prices': 'crop_prices.json',
        'reference': 'reference_data.json'
    }

    crop_yield_df = load_csv_file(paths['crop_yield'])
    soil_df = load_csv_file(paths['soil'])
    weather_df = load_csv_file(paths['weather'])
    market_df = load_csv_file(paths['market'])

    season_mapping = load_json_file(paths['season_mapping'])
    crop_prices = load_json_file(paths['crop_prices'])
    reference_data = load_json_file(paths['reference'])

    # If any of the critical CSVs are missing, return None to trigger upload UI
    if crop_yield_df is None or soil_df is None or weather_df is None or market_df is None:
        return None

    return {
        'crop_yield': crop_yield_df,
        'soil': soil_df,
        'weather': weather_df,
        'market': market_df,
        'season_mapping': season_mapping or {},
        'crop_prices': crop_prices or {},
        'reference': reference_data or {}
    }

@st.cache_data
def create_ml_dataset(data_dict):
    """Create a comprehensive dataset for ML model training"""
    crop_yield_df = data_dict['crop_yield'].copy()
    soil_df = data_dict['soil'].copy()
    weather_df = data_dict['weather'].copy()
    market_df = data_dict['market'].copy()

    # Ensure required columns exist
    for df, cols in [
        (crop_yield_df, ['State', 'District', 'Year', 'Crop', 'Yield_per_Hectare', 'Area_Hectares', 'Season']),
        (soil_df, ['State', 'District', 'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Moisture']),
        (weather_df, ['State', 'District', 'Year', 'Rainfall_mm', 'Temperature_C', 'Humidity']),
        (market_df, ['State', 'Year', 'Crop', 'Price_per_Quintal'])
    ]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in dataset: {missing}")

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

    # Fill missing numeric columns with medians
    num_cols = ml_data.select_dtypes(include=[np.number]).columns
    ml_data[num_cols] = ml_data[num_cols].fillna(ml_data[num_cols].median())

    # For categorical/text columns, fill with 'Unknown' or forward/backfill
    cat_cols = ml_data.select_dtypes(include=['object']).columns
    ml_data[cat_cols] = ml_data[cat_cols].fillna('Unknown')

    return ml_data

# ------------------------------
# CropRecommendationSystem class
# ------------------------------
class CropRecommendationSystem:
    def __init__(self):
        self.yield_model = None
        self.crop_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        # encoders
        self.state_encoder = LabelEncoder()
        self.district_encoder = LabelEncoder()
        self.season_encoder = LabelEncoder()

    def prepare_features(self, ml_data):
        """Prepare features for ML models"""
        # Create copies
        df = ml_data.copy()

        # Encode categorical variables (fit encoders here)
        df['State_encoded'] = self.state_encoder.fit_transform(df['State'].astype(str))
        df['District_encoded'] = self.district_encoder.fit_transform(df['District'].astype(str))
        df['Season_encoded'] = self.season_encoder.fit_transform(df['Season'].astype(str))

        # Feature columns for prediction
        feature_columns = [
            'State_encoded', 'District_encoded', 'Season_encoded', 'Year',
            'Area_Hectares', 'Rainfall_mm', 'Temperature_C', 'Humidity',
            'pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Organic_Carbon', 'Moisture'
        ]
        self.feature_columns = feature_columns

        # Ensure all feature columns exist
        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns after preparation: {missing}")

        return df[feature_columns], df['Yield_per_Hectare'], df['Crop'].astype(str)

    def train_models(self, ml_data, persist_path=None):
        """Train yield prediction and crop recommendation models"""
        X, y_yield, y_crop = self.prepare_features(ml_data)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode crop labels
        y_crop_encoded = self.label_encoder.fit_transform(y_crop)

        # Split data for yield and crop models
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

        # Optionally persist models & encoders
        if persist_path:
            with open(persist_path, 'wb') as f:
                pickle.dump({
                    'yield_model': self.yield_model,
                    'crop_model': self.crop_model,
                    'scaler': self.scaler,
                    'state_encoder': self.state_encoder,
                    'district_encoder': self.district_encoder,
                    'season_encoder': self.season_encoder,
                    'label_encoder': self.label_encoder,
                    'feature_columns': self.feature_columns
                }, f)

        return {
            'yield_r2': yield_r2,
            'yield_rmse': yield_rmse,
            'crop_accuracy': crop_accuracy
        }

    def load_models(self, persist_path):
        """Load pretrained models & encoders"""
        try:
            with open(persist_path, 'rb') as f:
                obj = pickle.load(f)
            self.yield_model = obj['yield_model']
            self.crop_model = obj['crop_model']
            self.scaler = obj['scaler']
            self.state_encoder = obj['state_encoder']
            self.district_encoder = obj['district_encoder']
            self.season_encoder = obj['season_encoder']
            self.label_encoder = obj['label_encoder']
            self.feature_columns = obj.get('feature_columns', self.feature_columns)
            return True
        except Exception:
            return False

    def predict_yield(self, input_features):
        """Predict crop yield based on input features"""
        if self.yield_model is None:
            return None

        features_scaled = self.scaler.transform([input_features])
        prediction = self.yield_model.predict(features_scaled)[0]
        return max(0.0, float(prediction))  # Ensure non-negative yield

    def recommend_crop(self, input_features, top_k=3):
        """Recommend best crop based on input features"""
        if self.crop_model is None:
            return None

        features_scaled = self.scaler.transform([input_features])

        # Get probabilities for all crops
        probabilities = self.crop_model.predict_proba(features_scaled)[0]
        # classifier.classes_ are encoded labels (ints). We take indices in that order.
        class_indices = self.crop_model.classes_.astype(int)
        # map classifier class index -> probability
        idx_prob_map = dict(zip(class_indices, probabilities))

        # get top k indices by probability (class indices)
        sorted_pairs = sorted(idx_prob_map.items(), key=lambda x: x[1], reverse=True)[:top_k]

        recommendations = []
        for class_idx, prob in sorted_pairs:
            # inverse transform to crop name (safe mapping)
            crop_name = self.label_encoder.inverse_transform([class_idx])[0]
            recommendations.append({'crop': crop_name, 'probability': float(prob)})

        return recommendations

    def calculate_profit(self, crop, yield_per_ha, area_hectares, data_dict):
        """Calculate expected profit for a crop"""
        crop_prices = data_dict.get('crop_prices', {})
        # price expected in INR per quintal
        price_per_quintal = float(crop_prices.get(crop, crop_prices.get(str(crop), 2000)))

        # yield_per_ha is in tonnes/ha, convert to quintals: 1 tonne = 10 quintals
        total_yield_tonnes = yield_per_ha * area_hectares
        total_yield_quintals = total_yield_tonnes * 10.0
        gross_revenue = total_yield_quintals * price_per_quintal

        # Estimate costs (simplified) - per hectare
        cost_per_hectare = {
            'Rice': 45000, 'Wheat': 40000, 'Cotton': 55000, 'Maize': 35000,
            'Soybean': 30000, 'Sugarcane': 80000, 'Barley': 32000, 'Mustard': 28000,
            'Gram': 25000, 'Peas': 30000, 'Fodder': 20000, 'Vegetables': 60000,
            'Watermelon': 40000, 'Cucumber': 45000
        }
        total_cost = cost_per_hectare.get(crop, 40000) * area_hectares
        profit = gross_revenue - total_cost
        roi = (profit / total_cost * 100) if total_cost > 0 else 0.0

        return {
            'revenue': float(gross_revenue),
            'cost': float(total_cost),
            'profit': float(profit),
            'roi': float(roi)
        }

# ------------------------------
# App main
# ------------------------------
def main():
    st.markdown("""
    <div class="main-header">
        <h1>🌾 AI-Powered Crop Cycle Planner Pro</h1>
        <p>Advanced Machine Learning for Smart Agricultural Decisions</p>
    </div>
    """, unsafe_allow_html=True)

    # Attempt to load canonical files
    with st.spinner("🔄 Loading agricultural datasets..."):
        data_dict = load_and_prepare_data()

    # If missing, ask user to upload required files (helpful during testing)
    if data_dict is None:
        st.warning("Some dataset files were not found on the server. Please upload required CSV/JSON files below (or place them in the app folder).")
        col1, col2 = st.columns(2)
        with col1:
            crop_yield_file = st.file_uploader("Upload crop_yield_data.csv", type=['csv'])
            soil_file = st.file_uploader("Upload soil_data.csv", type=['csv'])
            weather_file = st.file_uploader("Upload weather_data.csv", type=['csv'])
        with col2:
            market_file = st.file_uploader("Upload market_prices.csv", type=['csv'])
            season_map_file = st.file_uploader("Upload crop_season_mapping.json", type=['json'])
            crop_prices_file = st.file_uploader("Upload crop_prices.json", type=['json'])
            reference_file = st.file_uploader("Upload reference_data.json", type=['json'])

        if crop_yield_file and soil_file and weather_file and market_file:
            try:
                data_dict = {
                    'crop_yield': pd.read_csv(crop_yield_file),
                    'soil': pd.read_csv(soil_file),
                    'weather': pd.read_csv(weather_file),
                    'market': pd.read_csv(market_file),
                    'season_mapping': json.load(season_map_file) if season_map_file else {},
                    'crop_prices': json.load(crop_prices_file) if crop_prices_file else {},
                    'reference': json.load(reference_file) if reference_file else {}
                }
                st.success("Uploaded files loaded.")
            except Exception as e:
                st.error(f"Failed to read uploaded files: {e}")
                return
        else:
            st.info("Please upload the four CSV datasets to continue.")
            return

    # Create ML dataset and train models (or load persisted)
    persist_path = "crop_models.pkl"
    crop_system = CropRecommendationSystem()
    loaded = False

    # Try loading persisted models
    if st.sidebar.checkbox("Load previously trained models (if available)", value=True):
        loaded = crop_system.load_models(persist_path)
        if loaded:
            st.sidebar.success("Loaded saved models.")
        else:
            st.sidebar.info("No saved models available or failed to load.")

    if not loaded:
        with st.spinner("🧠 Preparing machine learning models and training..."):
            try:
                ml_data = create_ml_dataset(data_dict)
            except Exception as e:
                st.error(f"Error preparing ML dataset: {e}")
                return

            model_performance = crop_system.train_models(ml_data, persist_path=persist_path)
        st.success("✅ AI models trained successfully!")
    else:
        # if loaded, still need ml_data for insights below
        ml_data = create_ml_dataset(data_dict)

    # Display model performance (if available)
    st.markdown("---")
    if loaded:
        st.info("Models loaded from disk. Performance metrics shown are from when the models were trained previously.")
    try:
        if 'model_performance' not in locals():
            # If models loaded from file, compute a quick performance estimate (optional)
            model_performance = {'yield_r2': 0.0, 'yield_rmse': 0.0, 'crop_accuracy': 0.0}
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Yield Model R²</h4>
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
    except Exception:
        pass

    # ------------------------------
    # Sidebar inputs
    # ------------------------------
    st.sidebar.header("📋 Farm & Environmental Parameters")
    ref = data_dict.get('reference', {})

    states = ref.get('states', sorted(ml_data['State'].unique().tolist()))
    seasons = ref.get('seasons', sorted(ml_data['Season'].unique().tolist()))
    selected_state = st.sidebar.selectbox("State", states)
    available_districts = ref.get('districts', {}).get(selected_state, sorted(ml_data.loc[ml_data['State']==selected_state, 'District'].unique().tolist()))
    selected_district = st.sidebar.selectbox("District", available_districts if available_districts else ['Other'])

    area_hectares = st.sidebar.number_input("Farm Area (hectares)", 0.1, 1000.0, 2.0, step=0.1)
    season = st.sidebar.selectbox("Season", seasons)
    year = st.sidebar.number_input("Year", 2015, 2030, 2024)

    # Soil
    ph = st.sidebar.slider("Soil pH", 4.0, 9.0, 7.0, step=0.1)
    nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 150, 400, 275)
    phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 20, 80, 50)
    potassium = st.sidebar.slider("Potassium (kg/ha)", 150, 350, 250)
    organic_carbon = st.sidebar.slider("Organic Carbon (%)", 0.3, 1.5, 0.9, step=0.1)
    moisture = st.sidebar.slider("Soil Moisture (%)", 15, 45, 30)

    # Weather
    rainfall = st.sidebar.slider("Annual Rainfall (mm)", 300, 2000, 1000)
    temperature = st.sidebar.slider("Average Temperature (°C)", 15, 40, 27)
    humidity = st.sidebar.slider("Humidity (%)", 40, 85, 63)

    # Prediction button
    if st.sidebar.button("🚀 Generate AI Recommendations"):
        # Prepare input features robustly
        try:
            # Encode or fallback for unseen categories
            def safe_transform(encoder, value):
                if value in encoder.classes_:
                    return int(encoder.transform([value])[0])
                else:
                    # expand encoder classes to include the new value -> fallback to 0
                    return 0

            state_encoded = safe_transform(crop_system.state_encoder, str(selected_state))
            district_encoded = safe_transform(crop_system.district_encoder, str(selected_district))
            season_encoded = safe_transform(crop_system.season_encoder, str(season))

            input_features = [
                state_encoded, district_encoded, season_encoded, int(year), float(area_hectares),
                float(rainfall), float(temperature), float(humidity),
                float(ph), float(nitrogen), float(phosphorus),
                float(potassium), float(organic_carbon), float(moisture)
            ]

            # Get crop recommendations
            recommendations = crop_system.recommend_crop(input_features)
            if not recommendations:
                st.error("Crop recommendation model is not available.")
            else:
                st.header("🤖 AI-Generated Recommendations")

                # Display top recommendations
                for i, rec in enumerate(recommendations):
                    crop_name = rec['crop']
                    probability = rec['probability']

                    # Predict yield for this input (yield is crop-agnostic here; ideally you'd retrain per-crop or use crop as feature)
                    predicted_yield = crop_system.predict_yield(input_features) or 0.0

                    # Calculate profit
                    profit_info = crop_system.calculate_profit(
                        crop_name, predicted_yield, area_hectares, data_dict
                    )

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
                        fig = go.Figure(data=go.Bar(
                            x=['Revenue', 'Cost', 'Profit'],
                            y=[profit_info['revenue'], profit_info['cost'], profit_info['profit']]
                        ))
                        fig.update_layout(
                            title=f"{crop_name} Economics",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Seasonal analysis
                st.header("📅 Seasonal Analysis")
                season_crops = data_dict.get('season_mapping', {}).get(season, [])
                if season_crops:
                    st.info(f"**{season} Season Crops:** {', '.join(season_crops)}")
                    recommended_crops = [rec['crop'] for rec in recommendations]
                    seasonal_matches = set(recommended_crops) & set(season_crops)
                    if seasonal_matches:
                        st.success(f"✅ AI recommendations match seasonal crops: {', '.join(seasonal_matches)}")
                    else:
                        st.warning("⚠️ AI recommendations differ from traditional seasonal crops. Consider market opportunities!")
                else:
                    st.info("No season-to-crop mapping available for this season.")

                # Environmental suitability radar (normalize consistently 0-100)
                st.header("🌍 Environmental Suitability Analysis")
                factors = ['pH', 'Nitrogen', 'Phosphorus', 'Potassium', 'Rainfall', 'Temperature']
                # Normalize: pH scaled into 0-100 relative to 4-9; nutrients relative to chosen maxima
                ph_scaled = (ph - 4.0) / (9.0 - 4.0) * 100
                n_scaled = (nitrogen - 0) / (400 - 0) * 100
                p_scaled = (phosphorus - 0) / (100 - 0) * 100
                k_scaled = (potassium - 0) / (400 - 0) * 100
                rain_scaled = min(rainfall / 2000.0 * 100, 100)
                temp_scaled = (temperature - 0) / (40 - 0) * 100

                values = [ph_scaled, n_scaled, p_scaled, k_scaled, rain_scaled, temp_scaled]
                fig = go.Figure(data=go.Scatterpolar(
                    r=values,
                    theta=factors,
                    fill='toself',
                    name='Current Conditions'
                ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
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

    # ------------------------------
    # Data insights expander
    # ------------------------------
    with st.expander("📊 Agricultural Data Insights"):
        st.subheader("🌾 Crop Yield Comparison")
        try:
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
        except Exception as e:
            st.error(f"Failed to create yield chart: {e}")

        st.subheader("🗺️ Regional Analysis")
        try:
            state_yield = ml_data.groupby('State')['Yield_per_Hectare'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=state_yield.index,
                y=state_yield.values,
                title="Average Yield by State",
                labels={'x': 'State', 'y': 'Average Yield (tonnes/hectare)'}
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Failed to create state chart: {e}")

    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🌾 AI-Powered Crop Cycle Planner Pro | Advanced Agricultural Intelligence</p>
        <p><small>Empowering farmers with data-driven decisions for sustainable agriculture</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()