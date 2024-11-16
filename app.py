import os
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import zipfile


# Function to unzip files
def unzip_files(zip_path, extract_to):
    if not os.path.exists(extract_to):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped {zip_path} to {extract_to}")
    else:
        print(f"{extract_to} already exists. Skipping unzip.")

# Unzip storage and stock performance data
unzip_files('storage.zip', './storage')
unzip_files('stock_performance.zip', './stock_performance')

# Clean up ZIP files
os.remove('storage.zip')
os.remove('stock_performance.zip')


# Get API key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Global LLM and Embedding model
Settings.llm = OpenAI(model="gpt-4", max_tokens=6000)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

# Load the index
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# Create query engine
query_engine = index.as_query_engine(llm=Settings.llm)


# Streamlit app
st.title("AI Powered Financial Analyst")

# Create a simple text field for some information which tickers are supported
st.write("Supported tickers: AAPL, NVDA, GS, GOOGL, INTC")
st.write("For Historical Performance, please use the exact ticker symbol.")

report_type = st.selectbox(
    "What type of report do you want?",
    ("Single Stock Outlook", "Historical Performance"),
)


if report_type == "Single Stock Outlook":
    symbol = st.text_input("Stock Symbol")

    if symbol:
        with st.spinner(f"Generating report for {symbol}..."):
            response = query_engine.query(
                f"Write a report for {symbol} stock based on the news. Be sure to include potential risks and headwinds as well as opportunities."
            )

            st.write(response.response)

if report_type == "Historical Performance":
    symbol = st.text_input("Stock Symbol")

    if symbol:
        # Load the historical stock data from the corresponding CSV file
        csv_path = f"./stock_performance/{symbol}_prices_2024-01-01_to_2024-11-16.csv"
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)

            # Plotly Chart
            fig = go.Figure()

            # Add a line for the Closing Price
            fig.add_trace(
                go.Scatter(
                    x=data["Date"],
                    y=data["Close"],
                    mode="lines",
                    name="Closing Price",
                    line=dict(color="blue"),
                )
            )

            # Add a bar for Volume (on a secondary y-axis)
            fig.add_trace(
                go.Bar(
                    x=data["Date"],
                    y=data["Volume"],
                    name="Volume",
                    marker_color="orange",
                    opacity=0.6,
                    yaxis="y2",  # Associate Volume with the secondary y-axis
                )
            )

            # Customize the layout
            fig.update_layout(
                title=f"Historical Performance of {symbol}",
                xaxis_title="Date",
                yaxis=dict(
                    title="Price (USD)",  # Primary y-axis title
                    showgrid=True,
                ),
                yaxis2=dict(
                    title="Volume",  # Secondary y-axis title
                    overlaying="y",  # Overlay on the same x-axis
                    side="right",  # Position on the right side
                    showgrid=False,
                ),
                legend=dict(orientation="h", y=-0.2),
                height=500,
            )

            # Display the chart in Streamlit
            st.plotly_chart(fig, use_container_width=True)

            # Query the model for a textual summary
            with st.spinner(f"Generating report for {symbol}..."):
                response = query_engine.query(
                    f"Write a report on the historical performance of {symbol} stock year to date."
                )
                st.write(response.response)

        else:
            st.error(
                f"No historical data found for {symbol}. Please check the stock symbol."
            )
