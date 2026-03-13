# AI-Enabled Drone Analytics System 🚁📊

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://business-sem-4.streamlit.app/)

A comprehensive Streamlit dashboard for a **Business & Data Analysis Case Study** focusing on an AI-enabled Analytics System for drone imagery.

**Name**: Kishan Ojha  
**Course**: B.Tech CSE  
**Semester**: IV  
**Subject**: Business Studies  

---

## 🚀 Live Demo

You can view the live deployment of this project here:  
**[👉 https://business-sem-4.streamlit.app/](https://business-sem-4.streamlit.app/)**

---

## 📑 Project Sections

This dashboard provides five main sections for a complete business and data analysis:

### 1. 💰 Financial Analysis
Simulates the return on investment for an AI system.
* Calculates annual savings from operational cost reduction.
* Determines the Net Benefit and Payback Period based on an initial investment.

### 2. 📈 Data Overview & KPIs
Explores the generated synthetic data representing drone metrics.
* Displays a raw data table and statistical summaries.
* Includes a KPI dashboard showcasing average processing time, detection accuracy, defect rates, crop health, and client satisfaction.

### 3. 📊 Visualizations
Interactive charts to explore relationships between variables.
* **Plotly Bubble Chart:** Processing Time vs Satisfaction.
* **Seaborn Correlation Matrix:** Heatmap showing how features relate.
* Scatter plots and distribution curves to visually confirm business assumptions.

### 4. 🤖 Predictive Models
Applies Machine Learning to the drone analytics data.
* **Regression (Linear Regression):** Predicts Client Satisfaction using operational metrics.
* **Classification (Random Forest):** Predicts Crop Damage based on crop health scores and sensor telemetry.
* Includes an AI Image Defect Detection simulation generator.

### 5. 💡 Business Insights
Synthesizes the data into actionable business takeaways.
* Highlights time efficiency, cost savings, customer satisfaction improvements, and scalability.
* Concludes whether the company should formally adopt AI analytics based on the calculated ROI.

---

## 🛠️ Technologies Used

* **Python 3**
* **Streamlit** (Web Application Framework)
* **Pandas & NumPy** (Data Manipulation)
* **Scikit-Learn** (Machine Learning algorithms)
* **Matplotlib, Seaborn, Plotly** (Data Visualization)

---

## 💻 Running Locally

To run this application on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/Kishann911/Business-Sem-4.git
   cd Business-Sem-4
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your web browser and navigate to `http://localhost:8501/`
