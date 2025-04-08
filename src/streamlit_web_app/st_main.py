import streamlit as st
import os
import requests

BACKEND_URL = "http://inference:8000/process"
LABELS_MAPPING = {
    "Company Name": "Com",
    "Person": "P",
    "Money": "$",
    "Date": "Date",
    "Tenure": "Ten",
    "Country": "Co",
    "Street Address": "Addr",
    "STREET ADDRESS": "Addr",
    "State": "St",
    "City": "Ct",
    "Medicine Name": "Med",
    "ID": "ID",
    "Phone": "Ph"
}

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path

st.title("Omdena Legal Document Redaction")

# Initialize session state keys
if "processed" not in st.session_state:
    st.session_state.processed = False

if "last_uploaded_filename" not in st.session_state:
    st.session_state.last_uploaded_filename = None

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Check if it's a new file
    if uploaded_file.name != st.session_state.last_uploaded_filename:
        st.session_state.processed = False
        st.session_state.last_uploaded_filename = uploaded_file.name

    save_uploaded_file(uploaded_file, "output/input.pdf")

    if not st.session_state.processed:
        with st.spinner("Processing..."):
            with open("output/input.pdf", "rb") as f:
                files = {"file": f}
                response = requests.post(BACKEND_URL, files=files)

        if response.ok:
            result = response.json()
            st.session_state.result = result
            st.session_state.processed = True  # Mark as processed

    if st.session_state.processed:
        output_pdf_path = "output/Redacted.pdf"

        st.success("Processing complete!")

        with open(output_pdf_path, "rb") as file:
            st.download_button(
                label="Download Processed PDF",
                data=file,
                file_name="processed_output.pdf",
                mime="application/pdf"
            )


# Add legend section
with st.expander("Legend for Redaction Labels"):
    for label, abbreviation in LABELS_MAPPING.items():
        st.markdown(f"- **{label}**: `{abbreviation}`")