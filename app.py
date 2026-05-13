import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="AI Analytics Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Business Analytics Dashboard")
st.markdown("Upload your CSV file and get instant analytics insights.")

# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    # Read Dataset
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Uploaded Successfully!")

    # Dataset Preview
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head())

    # Dataset Info
    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())

    # Select Numeric Columns
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(numeric_cols) > 0:

        # KPI Section
        st.subheader("📈 KPI Metrics")

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.metric(
                "Total Value",
                round(df[numeric_cols[0]].sum(), 2)
            )

        with kpi2:
            st.metric(
                "Average Value",
                round(df[numeric_cols[0]].mean(), 2)
            )

        with kpi3:
            st.metric(
                "Maximum Value",
                round(df[numeric_cols[0]].max(), 2)
            )

        # Charts
        st.subheader("📊 Interactive Charts")

        chart_type = st.selectbox(
            "Select Chart Type",
            ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram"]
        )

        x_axis = st.selectbox("Select X-axis", df.columns)
        y_axis = st.selectbox("Select Y-axis", numeric_cols)

        if chart_type == "Bar Chart":
            fig = px.bar(df, x=x_axis, y=y_axis)

        elif chart_type == "Line Chart":
            fig = px.line(df, x=x_axis, y=y_axis)

        elif chart_type == "Scatter Plot":
            fig = px.scatter(df, x=x_axis, y=y_axis)

        elif chart_type == "Histogram":
            fig = px.histogram(df, x=y_axis)

        st.plotly_chart(fig, use_container_width=True)

        # Correlation Heatmap
        st.subheader("🔥 Correlation Heatmap")

        corr = df[numeric_cols].corr()

        heatmap = px.imshow(
            corr,
            text_auto=True,
            aspect="auto"
        )

        st.plotly_chart(heatmap, use_container_width=True)

        # AI Insights
        st.subheader("🤖 AI Insights")

        highest_col = df[numeric_cols].sum().idxmax()
        lowest_col = df[numeric_cols].sum().idxmin()

        st.info(f"Highest performing column: {highest_col}")
        st.warning(f"Lowest performing column: {lowest_col}")

        # Customer Segmentation
        if len(numeric_cols) >= 2:

            st.subheader("🧠 Customer Segmentation")

            cluster_data = df[numeric_cols[:2]].dropna()

            kmeans = KMeans(n_clusters=3, random_state=42)
            cluster_data["Cluster"] = kmeans.fit_predict(cluster_data)

            cluster_fig = px.scatter(
                cluster_data,
                x=numeric_cols[0],
                y=numeric_cols[1],
                color=cluster_data["Cluster"].astype(str),
                title="Customer Segments"
            )

            st.plotly_chart(cluster_fig, use_container_width=True)

        # Simple Forecasting
        if len(numeric_cols) >= 1:

            st.subheader("📉 Forecasting")

            forecast_df = df[[numeric_cols[0]]].dropna().reset_index()

            X = forecast_df.index.values.reshape(-1, 1)
            y = forecast_df[numeric_cols[0]].values

            model = LinearRegression()
            model.fit(X, y)

            future_days = np.array(
                range(len(X), len(X) + 10)
            ).reshape(-1, 1)

            predictions = model.predict(future_days)

            pred_df = pd.DataFrame({
                "Future_Index": future_days.flatten(),
                "Predicted_Value": predictions
            })

            forecast_fig = px.line(
                pred_df,
                x="Future_Index",
                y="Predicted_Value",
                title="Future Predictions"
            )

            st.plotly_chart(forecast_fig, use_container_width=True)

    else:
        st.error("No numeric columns found in dataset.")

else:
    st.info("Please upload a CSV file to begin analysis.")
