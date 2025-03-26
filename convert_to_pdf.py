import os
from docx2pdf import convert

# Define folders to search for .docx files
folders = ["Dissolution_Petitions", "Lease_Agreements", "NDA_Documents", "Partnership_Agreements"]

def convert_docx_to_pdf():
    """Converts all .docx files in the specified folders to .pdf format."""
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Skipping {folder}, it does not exist.")
            continue

        for file in os.listdir(folder):
            if file.endswith(".docx"):  # Only process .docx files
                docx_path = os.path.join(folder, file)
                pdf_path = os.path.join(folder, file.replace(".docx", ".pdf"))

                # Skip if PDF already exists
                if os.path.exists(pdf_path):
                    print(f"Skipping {pdf_path}, already exists.")
                    continue

                print(f"Converting: {docx_path} -> {pdf_path}")
                try:
                    convert(docx_path)  # Converts .docx to .pdf in the same folder
                    print(f"✔ Successfully converted: {pdf_path}")
                except Exception as e:
                    print(f"❌ Error converting {docx_path}: {e}")

if __name__ == "__main__":
    convert_docx_to_pdf()
