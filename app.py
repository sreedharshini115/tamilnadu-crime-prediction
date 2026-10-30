
import io
import json

import boto3
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

st.set_page_config(
    page_title="Tamil Nadu Crime Prediction",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: #e2e8f0;
}
[data-testid="stSidebar"] {
    background-color: #142035;
}
.block-container {
    padding-top: 2rem;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
[data-testid="stMetric"] {
    background: #19263c;
    padding: 18px;
    border: 1px solid #30415d;
    border-radius: 12px;
}
.stButton button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_dataset():
    data = pd.read_csv("tamilnadu_crime_sample.csv")

    required = [
        "City", "Year", "Crime_Type", "Population",
        "Crime_Count", "Latitude", "Longitude"
    ]

    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return data


df = load_dataset()


@st.cache_resource
def train_model():
    X = df[["City", "Year", "Crime_Type", "Population"]]
    y = df["Crime_Count"]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                ["City", "Crime_Type"]
            )
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessing),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ])

    pipeline.fit(X, y)
    return pipeline


model = train_model()


def create_pdf(prediction):
    buffer = io.BytesIO()

    document = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    content = [
        Paragraph(
            "Tamil Nadu Crime Prediction Report",
            styles["Title"]
        ),
        Spacer(1, 20)
    ]

    rows = [
        ["Field", "Value"],
        ["City", prediction["City"]],
        ["Year", str(prediction["Year"])],
        ["Crime Type", prediction["Crime_Type"]],
        ["Population", f'{prediction["Population"]:,}'],
        [
            "Predicted Crime Count",
            f'{prediction["Crime_Count"]:,.0f}'
        ],
        [
            "Predicted Crime Rate per 100k",
            f'{prediction["Crime_Rate"]:.2f}'
        ]
    ]

    table = Table(rows, colWidths=[210, 250])

    table.setStyle(TableStyle([
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.HexColor("#1d4ed8")
        ),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10)
    ]))

    content.append(table)
    content.append(Spacer(1, 20))

    content.append(Paragraph(
        "This report is based on a sample dataset. "
        "It is an educational machine learning demonstration, "
        "not an official crime forecast.",
        styles["BodyText"]
    ))

    document.build(content)
    buffer.seek(0)

    return buffer.getvalue()


st.title("Tamil Nadu Crime Prediction System")
st.write(
    "Machine Learning Based Crime Analytics and "
    "Prediction Dashboard"
)

st.info(
    "This application uses sample data for demonstration. "
    "Predictions are not official crime forecasts."
)

st.sidebar.header("Prediction Inputs")

city = st.sidebar.selectbox(
    "Select City",
    sorted(df["City"].dropna().unique())
)

year = st.sidebar.number_input(
    "Select Year",
    min_value=2020,
    max_value=2040,
    value=2027
)

crime_type = st.sidebar.selectbox(
    "Select Crime Type",
    sorted(df["Crime_Type"].dropna().unique())
)

population = st.sidebar.number_input(
    "Enter Population",
    min_value=1,
    value=1000000,
    step=10000
)

if st.sidebar.button("Predict Crime Count"):
    input_data = pd.DataFrame([{
        "City": city,
        "Year": int(year),
        "Crime_Type": crime_type,
        "Population": int(population)
    }])

    predicted_count = max(
        0.0,
        float(model.predict(input_data)[0])
    )

    crime_rate = predicted_count / population * 100000

    st.session_state["prediction"] = {
        "City": city,
        "Year": int(year),
        "Crime_Type": crime_type,
        "Population": int(population),
        "Crime_Count": predicted_count,
        "Crime_Rate": crime_rate
    }


tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Crime Map",
    "AI Assistant",
    "PDF Report"
])


with tab1:
    st.header("Crime Analytics Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", len(df))
    col2.metric("Total Cities", df["City"].nunique())
    col3.metric("Crime Types", df["Crime_Type"].nunique())
    col4.metric("Years", df["Year"].nunique())

    if "prediction" in st.session_state:
        result = st.session_state["prediction"]

        st.success(
            "Predicted Crime Count: "
            f'{result["Crime_Count"]:,.0f}'
        )

        st.metric(
            "Predicted Crime Rate per 100k",
            f'{result["Crime_Rate"]:.2f}'
        )

    left, right = st.columns(2)

    with left:
        city_data = (
            df.groupby("City", as_index=False)["Crime_Count"]
            .sum()
        )

        city_chart = px.bar(
            city_data,
            x="City",
            y="Crime_Count",
            color="City",
            title="Crime Count by City",
            template="plotly_dark"
        )

        st.plotly_chart(
            city_chart,
            use_container_width=True
        )

    with right:
        crime_data = (
            df.groupby("Crime_Type", as_index=False)["Crime_Count"]
            .sum()
        )

        crime_chart = px.pie(
            crime_data,
            names="Crime_Type",
            values="Crime_Count",
            title="Crime Type Distribution",
            template="plotly_dark"
        )

        st.plotly_chart(
            crime_chart,
            use_container_width=True
        )

    yearly_data = (
        df.groupby("Year", as_index=False)["Crime_Count"]
        .sum()
    )

    yearly_chart = px.line(
        yearly_data,
        x="Year",
        y="Crime_Count",
        markers=True,
        title="Year-wise Crime Trend",
        template="plotly_dark"
    )

    st.plotly_chart(
        yearly_chart,
        use_container_width=True
    )

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)


with tab2:
    st.header("Tamil Nadu Crime Map")

    map_data = (
        df.groupby("City", as_index=False)
        .agg({
            "Latitude": "mean",
            "Longitude": "mean",
            "Crime_Count": "sum"
        })
    )

    map_data = map_data.rename(columns={
        "Latitude": "lat",
        "Longitude": "lon"
    })

    map_data = map_data.dropna(subset=["lat", "lon"])

    st.map(
        map_data,
        latitude="lat",
        longitude="lon"
    )

    st.subheader("City-wise Crime Summary")
    st.dataframe(
        map_data,
        use_container_width=True
    )

    st.caption(
        "Map points represent city locations in the "
        "sample dataset, not real-time crime hotspots."
    )





with tab3:
    st.header("Crime Data Assistant")
    st.caption("Local CSV analytics assistant - No Bedrock required")

    st.write("Try asking:")
    st.write("Which city has the highest crime count?")
    st.write("Which crime type is most common?")
    st.write("Show year-wise crime trends")
    st.write("How many cities are in the dataset?")
    st.write("Explain my prediction")

    question = st.text_input(
        "Enter your question",
        key="local_ai_question"
    )

    if st.button("Ask Assistant", key="local_ai_button"):
        q = question.lower().strip()

        if not q:
            st.warning("Please enter a question.")

        elif "city" in q or "cities" in q:
            city_counts = (
                df.groupby("City")["Crime_Count"]
                .sum()
                .sort_values(ascending=False)
            )

            if "how many" in q or "number" in q:
                st.success(
                    f"The dataset contains {df['City'].nunique()} cities."
                )
            else:
                st.success(
                    f"Highest sample crime count: "
                    f"{city_counts.index[0]} "
                    f"({city_counts.iloc[0]:,.0f} cases)"
                )
                st.bar_chart(city_counts)

        elif "crime type" in q or "common" in q:
            crime_counts = (
                df.groupby("Crime_Type")["Crime_Count"]
                .sum()
                .sort_values(ascending=False)
            )

            st.success(
                f"Most common crime type in the sample: "
                f"{crime_counts.index[0]}"
            )
            st.bar_chart(crime_counts)

        elif "year" in q or "trend" in q:
            yearly = (
                df.groupby("Year")["Crime_Count"]
                .sum()
                .sort_index()
            )

            st.write("Year-wise sample crime counts:")
            st.line_chart(yearly)
            st.dataframe(yearly.rename("Crime Count"))

        elif "prediction" in q or "predict" in q:
            if "prediction" in st.session_state:
                p = st.session_state["prediction"]

                st.success(
                    f"Predicted crime count for {p['City']} "
                    f"in {p['Year']}: "
                    f"{p['Crime_Count']:,.0f}"
                )

                st.write(
                    "This is a machine learning estimate "
                    "based on the sample dataset. "
                    "It is not an official crime forecast."
                )
            else:
                st.info(
                    "Please generate a prediction "
                    "using the sidebar first."
                )

        elif "total" in q or "record" in q or "dataset" in q:
            st.success(
                f"The dataset contains {len(df)} records, "
                f"{df['City'].nunique()} cities, and "
                f"{df['Crime_Type'].nunique()} crime types."
            )

        else:
            st.info(
                "I can answer questions about cities, "
                "crime types, yearly trends, dataset "
                "statistics, and your latest prediction. "
                "Please ask one of these questions."
            )

    st.caption(
        "Answers are calculated from sample CSV data. "
        "This is a rule-based analytics assistant, "
        "not a generative language model."
    )


with tab4:
    st.header("Crime Prediction PDF Report")

    if "prediction" not in st.session_state:
        st.info(
            "Generate a prediction first using "
            "the sidebar."
        )
    else:
        result = st.session_state["prediction"]

        st.write("City:", result["City"])
        st.write("Year:", result["Year"])
        st.write("Crime Type:", result["Crime_Type"])
        st.write(
            "Predicted Crime Count:",
            f'{result["Crime_Count"]:,.0f}'
        )

        pdf_data = create_pdf(result)

        st.download_button(
            label="Download PDF Report",
            data=pdf_data,
            file_name="TamilNadu_Crime_Prediction_Report.pdf",
            mime="application/pdf"
        )

st.divider()
st.caption(
    "Tamil Nadu Crime Prediction System | "
    "Machine Learning Project"
)
