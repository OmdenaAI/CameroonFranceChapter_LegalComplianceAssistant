import streamlit as st
from main import run

def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


st.title("Omdena Legal Document Redaction")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    save_uploaded_file(uploaded_file, r"D:/Raghu Studies/omdena/CameroonFranceChapter_LegalComplianceAssistant/output/input.pdf")
    extracted_text = run()

    
    output_pdf_path = "output/Redacted.pdf"
    # Provide a download link for the processed PDF
    with open(output_pdf_path, "rb") as file:
        st.download_button(
            label="Download Processed PDF",
            data=file,
            file_name="processed_output.pdf",
            mime="application/pdf"
        )
    
    # Optionally display the extracted text
    st.subheader("Extracted Text")
    st.text_area("Text Preview", value=extracted_text, height=300)
    