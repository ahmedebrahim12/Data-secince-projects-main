import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import json
import plotly.graph_objects as go
import gdown
import os


# Load the pre-trained components
@st.cache_resource
def load_model_and_scaler():
    # model = load(r"random_forest_cluster.pkl")
    scaler = load(r"scaler.pkl")
    label_encoders = load(r"label_encoders.pkl")
    return model, scaler, label_encoders

def download_model():
    url = "https://drive.google.com/uc?id=1SN3RcdPWE3Omu65wtNk-OboEbYMjNyCc"
    output = "random_forest_cluster.pkl"
    if not os.path.exists(output):
        gdown.download(url, output, quiet=False)

download_model()

# ثم بعد التنزيل
model = load("random_forest_cluster.pkl")

rf_model, scaler, label_encoders = load_model_and_scaler()

with open("mappings.json", "r") as f:
    mappings = json.load(f)

df_final = pd.read_csv('labeled_data.csv')

x_cluster=df_final.drop(['Heart_Disease', 'Diabetes', 'Other_Cancer','Skin_Cancer', 'Arthritis', 'Depression','cluster','Height_(cm)','Weight_(kg)'],axis=1)

categorical_Binary=['Exercise','Sex', 'Smoking_History']
def preprocess_input(data):
    try:
        for col in categorical_Binary:
            data[col] = label_encoders[col].transform([data[col]])[0]
    except ValueError as e:
        st.error(f"Error during Binary Encoding: {e}")
        return None

    try:
        for col in data.columns:
            if col in mappings:
                data[col] = data[col].map(mappings[col])
            else:
                if col in ['General_Health', 'Checkup', 'Age_Category']:
                    st.warning(f"⚠️ Mapping or column '{col}' not found.")
    except Exception as e:
        st.error(f"Error during Ordinal Encoding: {e}")
        return None

    
    try:
        data=data.drop(['Height_(cm)','Weight_(kg)'],axis=1)
        data = scaler.transform(data)
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        st.write("🧩 Input columns before scaling:", data.columns.tolist())
        st.write("✅ Expected columns:", x_cluster.columns.tolist())
        return None


    feature_names = x_cluster.columns.to_list()
    data = pd.DataFrame(data, columns=feature_names)

    return data

st.title("🧬 Health Segmentation App")
st.write("""
### Classify your health profile and see your group statistics
Enter basic lifestyle and health info to see which group you belong to based on your habits and health indicators.
""")

st.sidebar.header("📝 Enter Your Health Information")

def user_input_features():
    sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
    age = st.sidebar.selectbox("Age Category", [
        '18-24', '25-29', '30-34', '35-39', '40-44', '45-49',
        '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80+'
    ])
    exercise = st.sidebar.selectbox("Do you exercise?", ["Yes", "No"])
    smoking = st.sidebar.selectbox("Smoking History", ["Yes", "No"])
    alcohol = st.sidebar.slider("Alcohol Consumption (per week)", 0.0, 30.0, 0.0)
    fruit = st.sidebar.slider("Fruit Consumption (per week)", 0.0, 60.0, 5.0)
    veg = st.sidebar.slider("Green Vegetable Consumption (per week)", 0.0, 45.0, 5.0)
    fries = st.sidebar.slider("Fried Potato Consumption (per week)", 0.0, 20.0, 5.0)
    height = st.sidebar.number_input("Height (cm)", 100.0, 250.0, 170.0)
    weight = st.sidebar.number_input("Weight (kg)", 30.0, 200.0, 70.0)
    general_health = st.sidebar.selectbox("General Health", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    checkup = st.sidebar.selectbox("Last Medical Checkup", [
        'Within the past year',
        'Within the past 2 years',
        'Within the past 5 years',
        '5 or more years ago',
        'Never'
    ])

    bmi = weight / ((height / 100) ** 2)

    input_dict = {
        'General_Health': general_health,
        'Checkup': checkup,
        'Exercise': exercise,
        'Sex': sex,
        'Age_Category': age,
        'Height_(cm)': height,
        'Weight_(kg)': weight,
        'BMI': round(bmi, 2),
        'Smoking_History': smoking,
        'Alcohol_Consumption': alcohol,
        'Fruit_Consumption': fruit,
        'Green_Vegetables_Consumption': veg,
        'FriedPotato_Consumption': fries
    }

    return pd.DataFrame(input_dict, index=[0])

data = user_input_features()

st.subheader("📋 User Input")
st.write(data)

preprocessed_data = preprocess_input(data)

disease_columns = ['Heart_Disease', 'Diabetes', 'Skin_Cancer', 'Arthritis', 'Depression']

if preprocessed_data is not None:
    if st.button("🔍 Predict My Health Group"):
        cluster_prediction = rf_model.predict(preprocessed_data)[0]
        st.success(f"✅ You belong to Health Group: **Group {cluster_prediction+1}**")

        # Filter only the predicted cluster
        cluster_data = df_final[df_final['cluster'] == cluster_prediction]

        # Prepare list of diseases with their Yes% in the predicted cluster
        disease_stats = []

        for disease in disease_columns:
            all_cluster_vals = df_final.groupby("cluster")[disease].value_counts(normalize=True).unstack().fillna(0)

            if disease == "Diabetes":
                for label_value, label_name in zip([1, 2], ["Yes", "Intermediate"]):
                    percent = cluster_data[disease].value_counts(normalize=True).get(label_value, 0) * 100
                    cluster_percents = all_cluster_vals.get(label_value, pd.Series([0]*len(all_cluster_vals))) * 100
                    sorted_clusters = cluster_percents.sort_values(ascending=False)
                    rank = sorted_clusters.index.get_loc(cluster_prediction) + 1
                    total_clusters = len(sorted_clusters)

                    if rank == 1:
                        note = f"🔺 Highest ({label_name})"
                    elif rank == total_clusters:
                        note = f"🔻 Lowest ({label_name})"
                    elif rank == 2:
                        note = f"🔸 2nd Highest ({label_name})"
                    elif rank == total_clusters - 1:
                        note = f"🔸 2nd Lowest ({label_name})"
                    else:
                        note = f"🔸 Rank {rank}/{total_clusters} ({label_name})"

                    disease_stats.append((f"{disease} - {label_name}", percent, note))
            else:
                percent = cluster_data[disease].value_counts(normalize=True).get(1, 0) * 100
                cluster_percents = all_cluster_vals.get(1, pd.Series([0]*len(all_cluster_vals))) * 100
                sorted_clusters = cluster_percents.sort_values(ascending=False)
                rank = sorted_clusters.index.get_loc(cluster_prediction) + 1
                total_clusters = len(sorted_clusters)

                if rank == 1:
                    note = "🔺 Highest among all groups"
                elif rank == total_clusters:
                    note = "🔻 Lowest among all groups"
                elif rank == 2:
                    note = "🔸 2nd Highest"
                elif rank == total_clusters - 1:
                    note = "🔸 2nd Lowest"
                else:
                    note = f"🔸 Rank {rank}/{total_clusters}"

                disease_stats.append((disease, percent, note))

        # Sort diseases by yes_percent descending
        disease_stats_sorted = sorted(disease_stats, key=lambda x: x[1], reverse=True)

        # Plot using Plotly
        fig = go.Figure()

        for disease, yes_percent, note in disease_stats_sorted:
            
            color = '#FF4C4C' if "Highest among all groups" in note else '#FF7F50'  # Red or Coral

            fig.add_trace(go.Bar(
                x=[yes_percent],
                y=[f"{disease} ({note})"],
                orientation='h',
                text=f"{yes_percent:.1f}%",
                textposition='auto',
                marker_color=color
            ))

        fig.update_layout(
            title=f"🩺 Probable disease incidence in Group {cluster_prediction+1}",
            xaxis_title="Percentage of People with Condition",
            yaxis=dict(autorange="reversed"),  # Highest on top
            height=400 + 50 * len(disease_columns),
        )

        st.plotly_chart(fig, use_container_width=True)
