# 🌾 AI-Based Crop Cycle Planner

An intelligent agricultural decision support system that uses machine learning to recommend optimal 3-season crop cycles, predict yields, and maximize farm profitability.

## 🎯 Project Overview

This final-year AI/ML project demonstrates:
- **Crop Recommendation**: ML-powered crop selection based on soil and weather conditions
- **Yield Prediction**: Accurate yield forecasting using environmental parameters
- **Profit Optimization**: ROI analysis and profitability calculations
- **Interactive Dashboard**: Professional Streamlit-based web interface

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation & Setup

1. **Clone/Download the project files**
   ```bash
   # Ensure you have all project files in the same directory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The app will automatically open at `http://localhost:8501`
   - If not, manually navigate to the URL shown in the terminal

## 📋 Features

### Core Functionality
- ✅ **Multi-Season Planning**: Kharif, Rabi, and Zaid season recommendations
- ✅ **Soil Analysis**: pH, N-P-K nutrient level optimization
- ✅ **Weather Integration**: Rainfall and temperature-based predictions
- ✅ **Yield Forecasting**: ML-based yield prediction (R² = 0.972)
- ✅ **Profit Analysis**: Comprehensive financial planning and ROI calculation

### Dashboard Features
- 📊 **Interactive Charts**: Yield comparison, profit analysis, ROI trends
- 📱 **Responsive Design**: Works on desktop, tablet, and mobile devices
- 🎨 **Professional UI**: Clean, farmer-friendly interface design
- 📈 **Real-time Updates**: Instant recommendations based on input changes
- 📋 **Comprehensive Reports**: Detailed season-wise crop plans

## 🧠 Machine Learning Models

### 1. Crop Recommendation Model
- **Algorithm**: Random Forest Classifier
- **Features**: pH, Nitrogen, Phosphorus, Potassium, Rainfall, Temperature
- **Output**: Top 3 suitable crops with confidence scores

### 2. Yield Prediction Model
- **Algorithm**: Random Forest Regressor
- **Features**: Crop type, soil parameters, weather conditions
- **Accuracy**: R² = 0.972
- **Output**: Yield per hectare in tonnes

### 3. Profit Calculation Engine
- **Input**: Yield predictions, market prices, cultivation costs
- **Output**: Gross revenue, net profit, ROI percentage

## 📁 Project Structure

```
crop-cycle-planner/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── crop_recommendation_model.pkl   # Trained crop recommendation model
├── yield_prediction_model.pkl      # Trained yield prediction model
├── crop_encoder.pkl               # Label encoder for crops
├── crop_encoder_yield.pkl         # Yield model crop encoder
├── crop_prices.json               # Market price data
├── crop_season_mapping.json       # Season-crop relationships
├── reference_data.json            # Location and crop reference data
├── soil_data.csv                  # Sample soil dataset
├── weather_data.csv               # Sample weather dataset
├── crop_yield_data.csv            # Sample yield dataset
└── market_prices.csv              # Sample price dataset
```

## 🎮 How to Use

1. **Launch the Application**
   ```bash
   streamlit run app.py
   ```

2. **Input Farm Details**
   - Select your state and district
   - Enter farm area in hectares
   - Adjust soil parameters (pH, N-P-K levels)
   - Set weather conditions (rainfall, temperature)

3. **Generate Recommendations**
   - Click "Generate Crop Plan" button
   - View AI-recommended 3-season crop cycle
   - Analyze yield predictions and profit forecasts

4. **Review Results**
   - Compare yields across seasons
   - Analyze profit vs cost breakdown
   - Review ROI trends and recommendations

## 📊 Sample Demo Data

The system includes comprehensive sample datasets:
- **Soil Data**: 1,600+ records across 8 states
- **Weather Data**: 3,400+ monthly records (2015-2024)
- **Crop Yield**: 4,300+ historical yield records
- **Market Prices**: 12,000+ price data points

## 💡 Key Innovations

1. **Multi-Objective Optimization**: Balances profit, yield, and soil health
2. **Season-Specific Recommendations**: Tailored suggestions for each growing season
3. **Real-time Processing**: Instant ML-based predictions
4. **Comprehensive Analysis**: Holistic view of farm profitability
5. **User-Friendly Interface**: Accessible to farmers with minimal technical knowledge

## 🎓 Academic Value

Perfect for final-year AI/ML project demonstrations:
- ✅ Real-world agricultural problem solving
- ✅ Multiple ML algorithms implementation
- ✅ Professional web application development
- ✅ Data visualization and analysis
- ✅ Complete end-to-end system design

## 📈 Performance Metrics

- **Crop Recommendation Accuracy**: 37% (realistic agricultural prediction)
- **Yield Prediction R²**: 0.972 (excellent prediction accuracy)
- **Processing Speed**: <5 seconds for complete analysis
- **User Interface**: Professional, responsive design
- **Data Coverage**: 8 states, 14 crops, 9+ years of data

## 🛠️ Technical Stack

- **Frontend**: Streamlit (Python web framework)
- **Backend**: Python with scikit-learn
- **ML Models**: Random Forest Classifier/Regressor
- **Visualization**: Plotly for interactive charts
- **Data Processing**: Pandas, NumPy
- **Deployment**: Local development, easily deployable to cloud

## 🎯 Future Enhancements

- Weather API integration for real-time data
- Satellite imagery analysis for crop monitoring
- Disease prediction and prevention recommendations
- Market price forecasting with time series analysis
- Mobile app development for field use

## 📞 Support & Documentation

For questions or issues:
1. Check the console output for error messages
2. Ensure all required files are in the same directory
3. Verify Python version (3.8+) and package installations
4. Review the sample input ranges in the sidebar

## 📄 License

This project is developed for educational purposes as a final-year AI/ML demonstration.

---

**🌾 Empowering Farmers with AI-Driven Agricultural Intelligence 🌾**
