import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


MAX_UPLOAD_SIZE_MB = 50

st.set_page_config(page_title="CSV Explorer",layout="wide",)
st.title("CSV Explorer")
st.caption("Inspect your dataset and discover its primary value patterns.")


with st.sidebar:
    st.header("Data Controls")

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"],
        help=f"CSV files up to {MAX_UPLOAD_SIZE_MB} MB are supported.",
    )


def load_csv(file):
    if file is None:
        return None, None

    if file.size > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        return None, f"The file is larger than {MAX_UPLOAD_SIZE_MB} MB."

    try:
        file_bytes = file.getvalue()
        dataframe = pd.read_csv(
            io.BytesIO(file_bytes),
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return None, "The CSV must use UTF-8 text encoding."
    except pd.errors.EmptyDataError:
        return None, "The uploaded CSV is empty."
    except pd.errors.ParserError as error:
        return None, f"The CSV format could not be parsed: {error}"
    except Exception as error:
        return None, f"The file could not be read: {error}"

    if dataframe.empty:
        return None, "The uploaded CSV has no data rows."

    if dataframe.shape[1] == 0:
        return None, "The uploaded CSV has no columns."

    return dataframe, None


dataframe, upload_error = load_csv(uploaded_file)

if upload_error:
    st.error(upload_error)

if dataframe is None:
    st.info("Upload a CSV file from the sidebar to begin exploring.")
    st.stop()


with st.sidebar:
    st.divider()

    selected_column = st.selectbox(
        "Attribute to Visualize",
        dataframe.columns,
    )


st.subheader("Dataset Overview")

metric_columns = st.columns(3)

metric_columns[0].metric(
    "Rows",
    f"{dataframe.shape[0]:,}",
)

metric_columns[1].metric(
    "Columns",
    f"{dataframe.shape[1]:,}",
)

metric_columns[2].metric(
    "Missing Values",
    f"{int(dataframe.isna().sum().sum()):,}",
)


st.markdown("#### Dataset Preview")

st.dataframe(
    dataframe.head(5),
    use_container_width=True,
    hide_index=True,
)


st.markdown("#### Column Metadata")

metadata = pd.DataFrame(
    {
        "Column": dataframe.columns,
        "Data Type": dataframe.dtypes.astype(str).values,
        "Missing Values": dataframe.isna().sum().values,
        "Non-Null Values": dataframe.notna().sum().values,
        "Unique Values": dataframe.nunique(dropna=True).values,
    }
)

st.dataframe(
    metadata,
    use_container_width=True,
    hide_index=True,
)


st.markdown("#### Numerical Summary")

numerical_data = dataframe.select_dtypes(include="number")

if numerical_data.empty:
    st.info("No numerical attributes were found in this dataset.")
else:
    numerical_summary = pd.DataFrame(
        {
            "Mean": numerical_data.mean(),
            "Median": numerical_data.median(),
            "Minimum": numerical_data.min(),
            "Maximum": numerical_data.max(),
        }
    ).round(3)

    st.dataframe(
        numerical_summary,
        use_container_width=True,
    )


st.divider()

st.subheader(f"Visual Analysis: {selected_column}")

selected_series = dataframe[selected_column]


if pd.api.types.is_numeric_dtype(selected_series):
    st.caption("Detected Attribute Type: Numerical")

    chart_data = selected_series.dropna()

    if chart_data.empty:
        st.warning("This attribute contains no values to plot.")
    else:
        figure, axis = plt.subplots(figsize=(10, 4.5))

        axis.hist(
            chart_data,
            bins="auto",
            color="#176B87",
            edgecolor="white",
        )

        axis.set_title(f"Distribution of {selected_column}")
        axis.set_xlabel(selected_column)
        axis.set_ylabel("Frequency")
        axis.grid(axis="y", alpha=0.25)

        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

else:
    st.caption("Detected Attribute Type: Categorical")

    category_counts = (
        selected_series
        .fillna("Missing")
        .astype(str)
        .value_counts()
        .head(30)
        .sort_values(ascending=True)
    )

    if category_counts.empty:
        st.warning("This attribute contains no values to plot.")
    else:
        figure, axis = plt.subplots(figsize=(10, 4.5))

        axis.barh(
            category_counts.index,
            category_counts.values,
            color="#E07A5F",
        )

        axis.set_title(f"Frequency of {selected_column}")
        axis.set_xlabel("Count")
        axis.set_ylabel(selected_column)
        axis.grid(axis="x", alpha=0.25)

        figure.tight_layout()
        st.pyplot(figure)
        plt.close(figure)

        if selected_series.nunique(dropna=True) > 30:
            st.caption("Showing the 30 most frequent categories.")