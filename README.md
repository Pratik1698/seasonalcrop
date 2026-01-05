# 🌾 Seasonal Crop Cycle Planner

A smart agricultural decision-support web application that helps farmers plan crops across Kharif, Rabi, and Zaid seasons based on soil, weather, and economic factors.
Built using Python and Streamlit, the system recommends the most profitable crop for each season.

## 📌 Features

🌱 Season-wise crop recommendations (Kharif, Rabi, Zaid)

📊 Yield prediction based on soil pH, nitrogen, rainfall, and temperature

💰 Profit estimation using crop prices and cultivation costs

🗺️ District-wise planning (Maharashtra)

📈 Visual profit comparison using charts

💾 Downloadable crop plan report (JSON)

## 🧠 How It Works

User enters farm details such as area, soil parameters, and weather conditions.

The system predicts crop yield using predefined agronomic rules.

Expected profit is calculated for each crop.

The most profitable crop is selected for each season.

Results are displayed with insights and charts.

## 🛠️ Technologies Used

Python

Streamlit

Pandas

NumPy

Matplotlib

JSON

Datetime

## 📂 Project Structure
- ### Seasonal-Crop-Cycle-Planner/
  -│
  -├── app.py -      # Main Streamlit application
  -├── README.md -     # Project documentation
  -├── requirements.txt  -   # Python dependencies

## ▶️ How to Run the Project

Clone the repository:

git clone 'https://github.com/your-username/seasonal-crop-cycle-planner.git'


Install dependencies:

'''bash
'pip install -r requirements.txt'


Run the app:
''' bash 
'streamlit run app.py'  
📸 Output Preview

Season-wise recommended crops

Expected yield and profit

Profit comparison bar chart

Downloadable JSON report

# 🚀 Future Enhancements

Integration with real-time weather APIs

Machine learning–based yield prediction

Fertilizer and irrigation recommendations

Mobile-friendly UI
