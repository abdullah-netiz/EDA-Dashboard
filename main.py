import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="EDA Interface",
    page_icon="📊",
    layout="wide"
)

st.title("Exploratory Data Analysis Interface")

with st.sidebar:
    st.header("Dataset Controls")
    f = st.file_uploader(
        "Upload a CSV file",
        type=["csv"]
    )

if f is None:
    st.info("Please upload a CSV file to begin.")
    st.stop()

try:
    df = pd.read_csv(f)
except Exception as e:
    st.error(f"Could not read the CSV file: {e}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV file is empty.")
    st.stop()

with st.sidebar:
    st.header("Attribute Selection")
    col = st.selectbox(
        "Select a column for visualization",
        df.columns
    )

st.header("Dataset Preview & Metadata")

st.subheader("First 5 Rows")
st.dataframe(
    df.head(),
    use_container_width=True,
    hide_index=True
)

st.subheader("Dataset Shape")

a, b = st.columns(2)
a.metric("Rows", df.shape[0])
b.metric("Columns", df.shape[1])

st.subheader("Column Data Types")

types = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str)
})

st.dataframe(
    types,
    use_container_width=True,
    hide_index=True
)

st.subheader("Missing Values per Column")

miss = pd.DataFrame({
    "Column": df.columns,
    "Missing Count": df.isna().sum(),
    "Missing Percentage": (
        df.isna().mean() * 100
    ).round(2)
})

st.dataframe(
    miss,
    use_container_width=True,
    hide_index=True
)

st.subheader("Numerical Summary")

num = df.select_dtypes(include="number")

if num.empty:
    st.info("No numerical columns found.")
else:
    summary = pd.DataFrame({
        "Mean": num.mean(),
        "Median": num.median(),
        "Minimum": num.min(),
        "Maximum": num.max()
    }).round(2)

    st.dataframe(
        summary,
        use_container_width=True
    )

st.header("Visualization")

s = df[col].dropna()

if pd.api.types.is_numeric_dtype(df[col]):
    st.subheader(f"Histogram of {col}")

    if s.empty:
        st.warning("This column has no values to display.")
    else:
        fig, ax = plt.subplots(figsize=(10, 5))

        counts, bins, bars = ax.hist(
            s,
            bins=20,
            density=True,
            color="#8ecae6",
            edgecolor="black",
            alpha=0.8,
            label="Histogram"
        )

        x = np.linspace(s.min(), s.max(), 300)
        bw = 1.06 * s.std() * len(s) ** (-1 / 5)

        if bw == 0 or np.isnan(bw):
            bw = 1

        y = np.exp(
            -0.5 * ((x[:, None] - s.values[None, :]) / bw) ** 2
        ).sum(axis=1)

        y = y / (len(s) * bw * np.sqrt(2 * np.pi))

        ax.plot(
            x,
            y,
            color="#e63946",
            linewidth=3,
            label="Density Line"
        )

        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()

        st.pyplot(fig)
        plt.close(fig)

else:
    st.subheader(f"Bar Chart of {col}")

    count = (
        df[col]
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(30)
    )

    percent = (count / count.sum() * 100).round(2)

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        count.index,
        count.values,
        color="#f4a261",
        edgecolor="black",
        label="Category Count"
    )

    ax.plot(
        count.index,
        count.values,
        color="#264653",
        marker="o",
        linewidth=3,
        label="Count Line"
    )

    ax.set_title(f"Frequency of {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    for bar, p in zip(bars, percent):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{p}%",
            ha="center",
            va="bottom"
        )

    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)