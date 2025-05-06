import math

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


st.title("Good Bad Ugly Wine")
st.write("Upload your wine quality dataset to analyze and predict quality")
# Read the csv data

ifile = st.file_uploader("Enter your csv file", type="csv", accept_multiple_files=False)

if ifile is not None:
    data = pd.read_csv(ifile)

    with st.expander("🔍 Data Overview", expanded=False):
        st.write("First 5 rows of data:")
        st.write(data.head())

        st.write("Data shape:", data.shape)

        if st.checkbox("Show detailed statistics"):
            st.write(data.describe())

    # Data Quality Check
    with st.expander("🧹 Data Quality Check", expanded=False):
        st.write("Missing values heatmap: ")
        fig, ax = plt.subplots()
        heat_map = sns.heatmap(data.isnull(), cbar=True, ax=ax)
        st.pyplot(plt)

    # Visualization
    with st.expander("📊 Visualizations", expanded=False):
        plot_options = st.multiselect("Select visualizations to display",
                      [
                          "Quality Distribution", "Citric Acid vs Quality",
                          "Volatile Acidity vs Quality", "Correlation Heatmap"
                      ],
                      default=["Quality Distribution","Correlation Heatmap"]
                      )
        if "Quality Distribution" in plot_options:
            st.write("### Quality Distribution")
            fig, ax = plt.subplots()
            sns.catplot(x="quality", data=data, kind="count")
            st.pyplot(plt)

        if "Citric Acid vs Quality" in plot_options:
            st.write("### Citric Acid vs Quality")
            fig, ax = plt.subplots()
            sns.barplot(x="quality", y="citric acid", data=data)
            st.pyplot(plt)

        if "Volatile Acidity vs Quality" in plot_options:
            st.write("### Volatile Acidity vs Quality")
            fig, ax = plt.subplots()
            sns.barplot(x="quality", y="volatile acidity", data=data)
            st.pyplot(plt)

        if "Correlation Heatmap" in plot_options:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots()
            correlation = data.corr()
            sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size': 8},cmap="coolwarm", ax=ax)
            st.pyplot(plt)

    with st.expander("🤖 Prediction Model", expanded=False):
        st.write("### Wine Quality prediction")

        if 'quality' not in data.columns:
            st.error("Error: The dataset must contain a 'quality' column for prediction.")
        else:

            # Model parameters
            col1, col2 = st.columns(2)
            with col1:
                test_size = st.slider("Test set size (%)", 10, 40, 20)
                n_estimators = st.slider("Number of trees", 50, 500, 100)

            with col2:
                max_depth = st.slider("Max tree depth", 2, 20, 10)
                random_state = st.number_input("Random state", 0, 100, 3)

            x = data.drop('quality', axis=1)
            y = data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0)

            # Train the data
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size/100, random_state=random_state)

            model = RandomForestClassifier()
            model.fit(x_train, y_train)

            x_test_prediction = model.predict(x_test)
            data_accuracy = accuracy_score(x_test_prediction, y_test)

            model_score = data_accuracy

            st.write(f"Prediction Accuracy = {math.floor(data_accuracy * 100)}%")

            st.write("### Feature Importance")
            feature_imp = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False)
            fig, ax = plt.subplots()
            sns.barplot(x=feature_imp, y=feature_imp.index, ax=ax)
            plt.xlabel('Feature Importance Score')
            plt.ylabel('Features')
            st.pyplot(plt)

else:
    st.info("Please upload a CSV file to analyze wine quality data")
    st.write("Expected format: CSV file with wine characteristics and a 'quality' column")