import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="AI-Enabled Drone Analytics", layout="wide")

st.title("AI-Enabled Drone Analytics System")
st.subheader("Business & Data Analysis Dashboard")
st.write("**Name**: Kishan Ojha | **Course**: B.Tech CSE | **Semester**: IV | **Subject**: Business Studies")
st.markdown("---")

# sidebar
st.sidebar.header("Navigation")
sections = ["Financial Analysis", "Data Overview & KPIs", "Visualizations", "Predictive Models", "Business Insights"]
selection = st.sidebar.radio("Go to", sections)

@st.cache_data
def load_data():
    np.random.seed(42)
    n = 500
    data = {
        "Images_Processed": np.random.randint(100, 1000, n),
        "Processing_Time_Hours": np.random.uniform(0.5, 5, n),
        "Detection_Accuracy": np.random.uniform(80, 98, n),
        "Crop_Health_Score": np.random.uniform(0.3, 0.9, n),
        "Defect_Detection_Rate": np.random.uniform(70, 95, n),
        "Sensor_Temperature": np.random.uniform(20, 45, n),
        "Client_Satisfaction": np.random.uniform(3, 5, n)
    }
    df = pd.DataFrame(data)
    df["Crop_Damage"] = np.where(df["Crop_Health_Score"] < 0.5, 1, 0)
    return df

df = load_data()

if selection == "Financial Analysis":
    st.header("1. Financial Analysis")
    current_operational_cost = 8000000
    investment = 7500000
    cost_reduction = 0.50

    annual_savings = current_operational_cost * cost_reduction
    net_benefit = annual_savings - investment
    payback_period = investment / annual_savings

    col1, col2, col3 = st.columns(3)
    col1.metric("Annual Savings", f"₹{annual_savings:,.0f}")
    col2.metric("Net Benefit (Year 1)", f"₹{net_benefit:,.0f}")
    col3.metric("Payback Period", f"~{payback_period:.1f} years")

    st.info("""
    **Financial Summary:**
    - AI reduces analysis time from 3–5 days to a few hours
    - Operational costs reduce by 50%
    """)

elif selection == "Data Overview & KPIs":
    st.header("2. Data Exploration & KPIs")
    st.dataframe(df.head())
    
    st.subheader("Statistical Summary")
    st.dataframe(df.describe())

    st.subheader("KPI Dashboard")
    avg_processing = df["Processing_Time_Hours"].mean()
    avg_accuracy = df["Detection_Accuracy"].mean()
    avg_defect = df["Defect_Detection_Rate"].mean()
    avg_crop = df["Crop_Health_Score"].mean()
    avg_satisfaction = df["Client_Satisfaction"].mean()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Avg Processing Time (hrs)", f"{avg_processing:.2f}")
    col2.metric("Avg Detection Accuracy (%)", f"{avg_accuracy:.2f}")
    col3.metric("Avg Defect Detection (%)", f"{avg_defect:.2f}")
    col4.metric("Avg Crop Health Score", f"{avg_crop:.2f}")
    col5.metric("Avg Client Satisfaction", f"{avg_satisfaction:.2f}")

    # Process time vs satisfaction
    fig, ax = plt.subplots(figsize=(10,6))
    names = ["Processing Time", "Detection Accuracy", "Defect Detection", "Crop Health", "Client Satisfaction"]
    values = [avg_processing, avg_accuracy, avg_defect, avg_crop, avg_satisfaction]
    ax.bar(names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax.set_title("Drone Analytics KPI Dashboard")
    ax.tick_params(axis='x', rotation=45)
    st.pyplot(fig)

elif selection == "Visualizations":
    st.header("3. Data Visualizations")
    
    st.subheader("Interactive Drone Analytics Dashboard (Plotly)")
    fig_plotly = px.scatter(
        df,
        x="Processing_Time_Hours",
        y="Client_Satisfaction",
        color="Detection_Accuracy",
        size="Images_Processed",
        hover_data=["Crop_Health_Score"],
        title="Processing Time vs Satisfaction (Size = Images Processed)"
    )
    st.plotly_chart(fig_plotly, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Correlation Matrix")
        fig1, ax1 = plt.subplots(figsize=(8,6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax1, fmt=".2f")
        st.pyplot(fig1)

        st.subheader("Detection Accuracy vs Satisfaction")
        fig3, ax3 = plt.subplots(figsize=(8,5))
        sns.scatterplot(x="Detection_Accuracy", y="Client_Satisfaction", data=df, ax=ax3, color="green")
        st.pyplot(fig3)

    with col2:
        st.subheader("Processing Time vs Satisfaction")
        fig2, ax2 = plt.subplots(figsize=(8,5))
        sns.scatterplot(x="Processing_Time_Hours", y="Client_Satisfaction", data=df, ax=ax2)
        st.pyplot(fig2)

        st.subheader("Defect Detection Rate Distribution")
        fig4, ax4 = plt.subplots(figsize=(8,5))
        sns.histplot(df["Defect_Detection_Rate"], bins=20, ax=ax4, kde=True, color="purple")
        st.pyplot(fig4)

elif selection == "Predictive Models":
    st.header("4. Machine Learning Models")

    st.subheader("Regression Model: Predicting Client Satisfaction")
    X_reg = df[["Processing_Time_Hours", "Detection_Accuracy", "Defect_Detection_Rate", "Crop_Health_Score"]]
    y_reg = df["Client_Satisfaction"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train_r, y_train_r)
    pred_r = model.predict(X_test_r)
    
    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{r2_score(y_test_r, pred_r):.4f}")
    col2.metric("Mean Squared Error (MSE)", f"{mean_squared_error(y_test_r, pred_r):.4f}")
    st.write("This model predicts client satisfaction based on drone analytics performance metrics.")

    st.markdown("---")

    st.subheader("Classification Model: AI Crop Damage Prediction")
    X_clf = df[["Detection_Accuracy", "Sensor_Temperature", "Crop_Health_Score"]]
    y_clf = df["Crop_Damage"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train_c, y_train_c)
    pred_c = rf.predict(X_test_c)

    acc = accuracy_score(y_test_c, pred_c)
    st.metric("Model Accuracy", f"{acc * 100:.2f}%")
    st.text("Classification Report:")
    st.text(classification_report(y_test_c, pred_c))
    st.write("This model simulates AI detecting crop damage from drone telemetry and health scores.")

    st.markdown("---")
    
    st.subheader("AI Defect Detection & Image Simulation")
    images_to_process = st.slider("Select number of images to simulate processing", 100, 1000, 500)
    probability = np.random.uniform(0.7, 0.95)
    defects = int(images_to_process * probability * 0.1)
    
    col3, col4 = st.columns(2)
    col3.metric("Images Processed", images_to_process)
    col4.metric("Detected Defects", defects)

    st.write("**Simulated Drone Aerial Image Analysis**")
    image_pixels = np.random.randint(0, 255, (100,100))
    fig_img, ax_img = plt.subplots(figsize=(4, 4))
    ax_img.imshow(image_pixels, cmap='gray' if np.random.rand() > 0.5 else 'viridis')
    ax_img.axis("off")
    st.pyplot(fig_img)

elif selection == "Business Insights":
    st.header("5. Business Insights & Conclusion")
    
    st.success("""
    ### Key Highlights
    - **Time Efficiency:** AI reduces analysis time from 3–5 days to a few hours.
    - **Cost Efficiency:** Operational costs reduce by 50%.
    - **Customer Satisfaction:** Increases rapidly due to faster reporting and high detection accuracy.
    - **Scalability:** Automation enables scaled operations for handling larger drone data volumes.
    """)

    st.info("""
    ### Conclusion & Recommendation
    The company should invest in **AI-enabled drone analytics**.
    - **Annual Savings**: ₹40 Lakhs
    - **Payback Period**: ~1.9 years
    
    Although the first year shows a negative net benefit due to the initial investment, the system becomes highly profitable after the payback period and provides continuous savings in future years. 
    Adopting AI analytics is a strategically beneficial decision for DataDrone Analytics Pvt. Ltd.
    """)
