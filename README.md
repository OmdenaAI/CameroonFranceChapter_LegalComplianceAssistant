# CameroonFranceChapter_LegalComplianceAssistant

# Models
Gliner is used for entity extraction on text based data 

YOLO model is used for object detection on Signature and Logo

# Steps to run the application
1) First inside the src/model_inferencing directory run 
     1.1) git clone https://github.com/ultralytics/yolov5
     1.2) create a new folder runs/train/exp/weights and paste the weights best.pt and last.pt
     1.3) Drive link of YOLO fine tuned model: https://drive.google.com/drive/folders/1HYh70f5pWsUXggeNHLMTCU8mTP4ccTpc?usp=drive_link

2) Create a new folder inside src/model_inferencing directory called gliner_inference_model and paste the contents from drive link into this directory
    Drive link: https://drive.google.com/file/d/1e2sKYOgQS3a1LM-xuT0JL2EzQJfGDv7K/view?usp=drive_link
3) Create a new output folder in root directory
4) Install Docker desktop in your machine 
5) Finally in the root folder, run the following command 
    docker-compose up --build
6) The streamlit app runs in localhost:8501 and the user can upload the file
7) The backend api runs which does model inferencing runs in localhost:8000 port
8) To shutdown the , docker-compose down
