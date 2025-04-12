#!/usr/bin/env python
# coding: utf-8

# # 1. Define the Database Schema to store PII information

# In[1]:


# create a CSV file to store PII information


# In[62]:


# create a table called pii_records to store:

# record_number (INTEGER, Primary Key)

# filename (TEXT)

# pii_type (TEXT)

# pii_value (TEXT)

# confidence (REAL)

# Sample Data (in pii_masked_data.csv):
# record_number,filename,pii_type,pii_value,masked,date
# 1001,hr_records.csv,Email,john.doe@example.com,y,4/10/2025
# 1002,TUXZt20, finance_2023.xlsx,SSN,123-45-6789,y,4/7/2025 
# 1003,logs.txt,Phone,555-123-4567,y,4/6/2025
# 1004,hr_records.csv,Email,jane.doe@example.com,y,4/8/2025
# 1005,hr_record1.pdf, name, Mike,n,4/8/2025
# 1006,hr_record1.pdf, address, 405 Hilgard Avenue Los Angeles CA 90095,n,4/8/2025
# 1007, hr_record1.pdf,signature,picture1,y,4/8/2025


# # 2. The Dashboard for AI Legal Redaction from CSV

# In[64]:


import plotly.express as px 
import ipywidgets as widgets
from IPython.display import display, Markdown

# Load CSV
df = pd.read_csv("pii_masked_data.csv")

# Title
display(Markdown("## 🔐 PII Redaction Dashboard"))

# 1. Bar Chart: Masked vs Unmasked
masked_summary = df["masked"].value_counts().rename({1: "Masked", 0: "Unmasked"}).reset_index()
masked_summary.columns = ["Status", "Count"]
fig1 = px.bar(
    masked_summary,
    x='Status', y='Count',
    title="🛡️ Masked vs Unmasked Records",
    color='Status',  # trigger color assignment
    color_discrete_sequence=["#1f77b4"] * len(masked_summary)
)
fig1.show()

# 2. Bar Chart: PII Type Distribution
pii_counts = df["pii_type"].value_counts().reset_index()
pii_counts.columns = ["PII Type", "Count"]
fig2 = px.bar(
    pii_counts,
    x="PII Type", y="Count",
    title="🔍 Count of PII Types",
    color="PII Type",
    color_discrete_sequence=["#8B4513"] * len(pii_counts)
)
fig2.show()

# 3. Records per File
file_counts = df["filename"].value_counts().reset_index()
file_counts.columns = ["Filename", "Count"]
fig3 = px.bar(
    file_counts,
    x="Filename", y="Count",
    title="📁 Records per File",
    color="Filename",
    color_discrete_sequence=["#2E8B57"] * len(file_counts)
)
fig3.show()


# In[54]:


import plotly.express as px

# Prepare data
masked_summary = df["masked"].value_counts().rename({1: "Masked", 0: "Unmasked"}).reset_index()
masked_summary.columns = ["Status", "Count"]

# Pie chart with green shades
fig_pie = px.pie(
    masked_summary,
    names="Status",
    values="Count",
    title="🟢 Masked vs Unmasked Records (Pie Chart)",
    color_discrete_sequence=["#2E8B57", "#98FB98"]  # Two shades of green
)

fig_pie.update_traces(textposition='inside', textinfo='percent+label')
fig_pie.show()


# In[55]:


# Dropdown filter
pii_selector = widgets.Dropdown(
    options=df["pii_type"].unique(),
    description='Filter by PII Type:',
    style={'description_width': 'initial'}
)

def display_filtered_pii(pii_type):
    display(Markdown(f"### Records with PII Type: `{pii_type}`"))
    display(df[df["pii_type"] == pii_type])

widgets.interact(display_filtered_pii, pii_type=pii_selector)


# In[58]:


print(df.columns)


# In[60]:


df.columns = df.columns.str.strip().str.lower()


# In[61]:


import plotly.express as px
import pandas as pd

# Convert date column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group by date to count redactions
redactions_over_time = df.groupby('date').size().reset_index(name='Redaction Count')

# Create the green time series chart
fig_time_series = px.line(
    redactions_over_time,
    x='date',
    y='Redaction Count',
    title="📆 Time Series of PII Redactions",
    markers=True,
    line_shape="linear",
    color_discrete_sequence=["green"]
)

fig_time_series.update_layout(xaxis_title="Date", yaxis_title="Number of Redactions")
fig_time_series.show()


# # 3. The Dashboard for AI Legal Redaction with Streamlit

# In[31]:


import streamlit as st
import pandas as pd

# Load CSV
df = pd.read_csv("pii_masked_data.csv")

# Title
st.title("🔐 PII Masking Summary Dashboard")

# Metrics
total_records = len(df)
masked_count = df[df['masked'] == 'y'].shape[0]
unmasked_count = df[df['masked'] == 'n'].shape[0]

# Top KPIs
col1, col2, col3 = st.columns(3)
col1.metric("📄 Total PII Records", total_records)
col2.metric("✅ Masked", masked_count)
col3.metric("❌ Unmasked", unmasked_count)

# Chart: PII Types
st.subheader("📊 PII Types Count")
pii_counts = df['pii_type'].value_counts()
st.bar_chart(pii_counts)

# Chart: Masked vs Unmasked
st.subheader("🛡️ Masking Status")
masking_summary = df['masked'].value_counts().rename(index={'y': 'Masked', 'n': 'Unmasked'})
st.bar_chart(masking_summary)

# Filter Options
st.subheader("🔍 Filter Records")

filename_filter = st.selectbox("Select Filename", options=['All'] + list(df['filename'].unique()))
pii_type_filter = st.selectbox("Select PII Type", options=['All'] + list(df['pii_type'].unique()))

filtered_df = df.copy()
if filename_filter != 'All':
    filtered_df = filtered_df[filtered_df['filename'] == filename_filter]
if pii_type_filter != 'All':
    filtered_df = filtered_df[filtered_df['pii_type'] == pii_type_filter]

# Display Filtered Data
st.dataframe(filtered_df)

# Footer
st.caption("🧠 Powered by Streamlit")


# # 4. The Dashboard for AI Legal Redaction from sqlite3

# In[59]:


import sqlite3
import pandas as pd

# Load sample data
csv_path = "pii_masked_data.csv"
df = pd.read_csv(csv_path)

# Connect to SQLite DB (or create it)
conn = sqlite3.connect("pii_redaction_logs.db")
cursor = conn.cursor()

# Create the pii_records table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS pii_records (
        record_number INTEGER PRIMARY KEY,
        filename TEXT,
        pii_type TEXT,
        pii_value TEXT,
        marked BOOLEAN,
        date TEXT
    )
""")

# Insert data from CSV into table
df.to_sql("pii_records", conn, if_exists="replace", index=False)

# Display what was inserted
result = pd.read_sql_query("SELECT * FROM pii_records", conn)
print(result)

# Clean up
conn.commit()
conn.close()


# In[30]:


import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(page_title="PII Type Summary", layout="centered")

st.title("🔐 PII Type Summary Dashboard")

# Connect to SQLite database
conn = sqlite3.connect("pii_redaction_logs.db")
df = pd.read_sql_query("SELECT * FROM pii_records", conn)
conn.close()

# Show raw data
with st.expander("📄 View Raw Data"):
    st.dataframe(df)

# Group and count by pii_type
st.subheader("📊 Count of Each PII Type Detected")
pii_counts = df['pii_type'].value_counts().sort_values(ascending=False)

# Display as table and chart
st.write("### PII Type Counts Table")
st.table(pii_counts)

st.write("### PII Type Bar Chart")
st.bar_chart(pii_counts)

# Optional filter by file or redaction status
if st.checkbox("Filter by Filename"):
    file_selected = st.selectbox("Select a file", df["filename"].unique())
    filtered = df[df["filename"] == file_selected]
    st.write(filtered[["record_number", "pii_type", "pii_value", "masked", "date"]])


# In[ ]:




