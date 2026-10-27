import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import speech_recognition as sr
from io import BytesIO

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="AI Analytics Dashboard",
    layout="wide"
)

# -----------------------------------
# DARK THEME
# -----------------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1,h2,h3,h4 {
    color: #00FFAA;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TITLE
# -----------------------------------
st.title("🚀 AI Powered Data Analytics Dashboard")

# -----------------------------------
# LOGIN SYSTEM
# -----------------------------------
st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if username == "admin" and password == "1234":

    st.sidebar.success("✅ Login Successful")

    # -----------------------------------
    # FILE UPLOAD
    # -----------------------------------
    file = st.file_uploader(
        "📂 Upload CSV or Excel File",
        type=["csv", "xlsx"]
    )

    if file is not None:

        try:

            # -----------------------------------
            # READ FILE
            # -----------------------------------
            if file.name.endswith(".csv"):
                df = pd.read_csv(file)

            else:
                df = pd.read_excel(file)

            # -----------------------------------
            # DATA PREVIEW
            # -----------------------------------
            st.subheader("📌 Dataset Preview")
            st.dataframe(df.head())

            # -----------------------------------
            # DATA INFO
            # -----------------------------------
            st.subheader("📊 Dataset Information")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Rows", len(df))
            c2.metric("Columns", len(df.columns))
            c3.metric("Missing Values", int(df.isnull().sum().sum()))
            c4.metric("Duplicates", int(df.duplicated().sum()))

            # -----------------------------------
            # MISSING VALUE ANALYSIS
            # -----------------------------------
            st.subheader("🧠 Missing Value Analysis")

            missing = df.isnull().sum()
            missing = missing[missing > 0]

            if len(missing) > 0:

                st.dataframe(missing)

                fig1, ax1 = plt.subplots(figsize=(8,4))

                missing.plot(
                    kind="bar",
                    ax=ax1
                )

                ax1.set_title("Missing Values Chart")

                st.pyplot(fig1)

            else:
                st.success("✅ No Missing Values")

            # -----------------------------------
            # CLEAN DATA
            # -----------------------------------
            st.subheader("🧹 Data Cleaning")

            df = df.drop_duplicates()

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns

            categorical_cols = df.select_dtypes(
                exclude=np.number
            ).columns

            # Fill Numeric Missing
            for col in numeric_cols:
                df[col] = df[col].fillna(
                    df[col].mean()
                )

            # Fill Categorical Missing
            for col in categorical_cols:
                df[col] = df[col].fillna(
                    "Unknown"
                )

            st.success("✅ Cleaning Completed")

            # -----------------------------------
            # FILTER SECTION
            # -----------------------------------
            st.sidebar.subheader("🎛 Filters")

            if len(categorical_cols) > 0:

                filter_col = st.sidebar.selectbox(
                    "Select Filter Column",
                    categorical_cols
                )

                filter_value = st.sidebar.selectbox(
                    "Select Value",
                    df[filter_col].unique()
                )

                df = df[
                    df[filter_col] == filter_value
                ]

            # -----------------------------------
            # KPI SELECTION
            # -----------------------------------
            if len(numeric_cols) > 0:

                st.subheader("🎯 KPI Selection")

                kpi = st.selectbox(
                    "Select KPI",
                    numeric_cols
                )

                st.success(f"Selected KPI: {kpi}")

                # -----------------------------------
                # CHART TYPE
                # -----------------------------------
                chart_type = st.selectbox(
                    "📊 Select Chart Type",
                    [
                        "Line",
                        "Bar",
                        "Histogram",
                        "Scatter"
                    ]
                )

                # -----------------------------------
                # CHART VISUALIZATION
                # -----------------------------------
                st.subheader("📈 KPI Visualization")

                fig2, ax2 = plt.subplots(
                    figsize=(10,5)
                )

                if chart_type == "Line":

                    df[kpi].plot(
                        ax=ax2,
                        marker='o'
                    )

                elif chart_type == "Bar":

                    df[kpi].head(10).plot(
                        kind="bar",
                        ax=ax2
                    )

                elif chart_type == "Histogram":

                    df[kpi].hist(
                        ax=ax2
                    )

                elif chart_type == "Scatter":

                    ax2.scatter(
                        range(len(df)),
                        df[kpi]
                    )

                ax2.set_title(f"{kpi} Chart")

                st.pyplot(fig2)

                # -----------------------------------
                # HEATMAP
                # -----------------------------------
                st.subheader("🔥 Correlation Heatmap")

                fig3, ax3 = plt.subplots(
                    figsize=(10,6)
                )

                sns.heatmap(
                    df.corr(numeric_only=True),
                    annot=True,
                    cmap="coolwarm",
                    ax=ax3
                )

                st.pyplot(fig3)

                # -----------------------------------
                # AI INSIGHTS
                # -----------------------------------
                st.subheader("🤖 AI Insights")

                avg = round(df[kpi].mean(),2)
                mx = round(df[kpi].max(),2)
                mn = round(df[kpi].min(),2)
                std = round(df[kpi].std(),2)

                a1, a2, a3, a4 = st.columns(4)

                a1.metric("Average", avg)
                a2.metric("Maximum", mx)
                a3.metric("Minimum", mn)
                a4.metric("Std Dev", std)

                # -----------------------------------
                # TREND DETECTION
                # -----------------------------------
                st.subheader("📉 Trend Detection")

                if df[kpi].iloc[-1] > df[kpi].iloc[0]:
                    st.success("📈 Increasing Trend")

                elif df[kpi].iloc[-1] < df[kpi].iloc[0]:
                    st.error("📉 Decreasing Trend")

                else:
                    st.info("➖ Stable Trend")

                # -----------------------------------
                # ANOMALY DETECTION
                # -----------------------------------
                st.subheader("🚨 Anomaly Detection")

                mean = df[kpi].mean()
                std = df[kpi].std()

                anomalies = df[
                    (df[kpi] > mean + 2*std) |
                    (df[kpi] < mean - 2*std)
                ]

                st.write(
                    "Anomalies Found:",
                    len(anomalies)
                )

                if not anomalies.empty:
                    st.dataframe(anomalies)

                # -----------------------------------
                # PREDICTION
                # -----------------------------------
                st.subheader("🔮 Predictive Analytics")

                X = np.arange(len(df)).reshape(-1,1)
                y = df[kpi]

                model = LinearRegression()
                model.fit(X, y)

                future = model.predict(
                    [[len(df)+1]]
                )

                st.success(
                    f"Next Prediction: {round(future[0],2)}"
                )

                # -----------------------------------
                # PREDICTION GRAPH
                # -----------------------------------
                predictions = model.predict(X)

                fig4, ax4 = plt.subplots()

                ax4.plot(
                    y.values,
                    label="Actual"
                )

                ax4.plot(
                    predictions,
                    label="Predicted"
                )

                ax4.legend()

                st.pyplot(fig4)

                # -----------------------------------
                # WHAT IF ANALYSIS
                # -----------------------------------
                st.subheader("🧪 What-If Analysis")

                increase = st.slider(
                    "Increase KPI %",
                    0,
                    100,
                    10
                )

                modified = df[kpi] * (
                    1 + increase/100
                )

                st.dataframe(modified)

                # -----------------------------------
                # TOP & BOTTOM
                # -----------------------------------
                st.subheader("🏆 Top & Bottom 5")

                b1, b2 = st.columns(2)

                with b1:
                    st.write("### Top 5")
                    st.dataframe(
                        df.nlargest(5, kpi)
                    )

                with b2:
                    st.write("### Bottom 5")
                    st.dataframe(
                        df.nsmallest(5, kpi)
                    )

                # -----------------------------------
                # CATEGORY SUMMARY
                # -----------------------------------
                if len(categorical_cols) > 0:

                    st.subheader("📋 Category Summary")

                    category = st.selectbox(
                        "Select Category",
                        categorical_cols
                    )

                    summary = df.groupby(
                        category
                    )[numeric_cols].mean()

                    st.dataframe(summary)

                # -----------------------------------
                # AI CHAT
                # -----------------------------------
                st.subheader("💬 AI Chat")

                question = st.text_input(
                    "Ask Question"
                )

                if question:

                    q = question.lower()

                    if "average" in q:
                        st.write(avg)

                    elif "max" in q:
                        st.write(mx)

                    elif "min" in q:
                        st.write(mn)

                    elif "sum" in q:
                        st.write(df[kpi].sum())

                    else:
                        st.warning(
                            "Try asking average, max, min, sum"
                        )

                # -----------------------------------
                # VOICE QUERY
                # -----------------------------------
                st.subheader("🎤 Voice Assistant")

                if st.button("Speak"):

                    r = sr.Recognizer()

                    try:

                        with sr.Microphone() as source:

                            st.write("Speak Now...")
                            audio = r.listen(source)

                        text = r.recognize_google(audio)

                        st.success(
                            f"You Said: {text}"
                        )

                    except:
                        st.error(
                            "Voice Error"
                        )

                # -----------------------------------
                # DOWNLOAD CSV
                # -----------------------------------
                st.subheader("⬇ Download Report")

                csv = df.to_csv(
                    index=False
                )

                st.download_button(
                    "Download Cleaned CSV",
                    csv,
                    "cleaned_data.csv",
                    "text/csv"
                )

                # -----------------------------------
                # DOWNLOAD EXCEL
                # -----------------------------------
                output = BytesIO()

                with pd.ExcelWriter(
                    output,
                    engine='xlsxwriter'
                ) as writer:

                    df.to_excel(
                        writer,
                        index=False
                    )

                st.download_button(
                    "📥 Download Excel Report",
                    output.getvalue(),
                    "report.xlsx"
                )

                # -----------------------------------
                # DATA HEALTH
                # -----------------------------------
                st.subheader("❤️ Data Health")

                health = {
                    "Rows": len(df),
                    "Columns": len(df.columns),
                    "Numeric": list(numeric_cols),
                    "Categorical": list(categorical_cols)
                }

                st.write(health)

            else:
                st.error("❌ No Numeric Columns")

        except Exception as e:

            st.error(f"Error: {e}")

else:
    st.sidebar.warning(
        "Use Username = admin & Password = 1234"
    )