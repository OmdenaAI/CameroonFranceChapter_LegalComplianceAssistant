# Document Upload Streamlit App

This is a simple Streamlit app that allows users to upload documents and view their content. The app reads the file's content and displays it on the page. The app allows you to redact personally identifiable information (PII).

## Features

- Upload `.txt`, `.pdf`, `.jpg`, `.jpeg`, `.png`, `.csv`, `.pptx`, or `.docx` files.
- View the content of the uploaded document directly within the app.
- Redact PII.

#### Supported File Types:
- **Text Files (.txt)**: View the plain text content of the file.
- **PDF Files (.pdf)**: Extract and view the text content of the PDF.
- **Word Documents (.docx)**: View the text from a Word document.
- **CSV Files (.csv)**: View the contents of a CSV file as a table.
- **Image Files (.jpg, .jpeg, .png, .gif)**: Display the uploaded image.
- **PowerPoint Presentations (.pptx)**: Extract and view the text from slides.

## Requirements

You can install the required packages using `pip`:
```
pip install -r requirements.txt
```

## How to Run the App

    Clone or download this repository.

    Navigate to the folder containing the app.py file.

    Run the following command to start the Streamlit app:

    ```
    streamlit run app.py
    ```

    Once the app is running, open your browser and go to http://localhost:8501.

## File Handling

    Text Files (.txt): Content is read directly from the file and displayed.
    PDF Files (.pdf): Content is extracted using the PyPDF2 library.
    Word Documents (.docx): Content is extracted using the python-docx library.

## License

This project is open-source and available under the [MIT License](https://opensource.org/license/mit).