# Disaster-Response-Pipeline

## Project Overview
The goal of this project is to use supervised learning techniques to classify messages during disaster response campaigns. The data that Figure Eight provided was used to train the machine learning model. A flask application was built to classify new messages and display the results. Some extra graphs were included to show the contents of the data provided.

## File Contents
The project contains the following files

    .
    ├── Notebook Files
        ├── DRData.db
        ├── ETL Pipeline Preparation.ipynb
        ├── ML Pipeline Preparation.ipynb
        └── classifier.pkl
    ├── app     
    │   ├── templates   
    │       ├── go.html                      # Classification result page of web app
    │       └── master.html                  # Main page of web app    
    │   └── run.py                           # Flask file that runs app
    ├── data                   
    │   ├── DisasterResponse.db              # Database to save clean data to  
    │   ├── disaster_categories.csv          # Categories data to process
    │   ├── disaster_messages.csv            # Messages data to process
    │   └── process_data.py                  # ETL pipeline script to clean data
    ├── models
    │   ├── classifier.pkl                   # Saved Model  
    │   └── train_classifier.py              # Train ML model        
    ├── LICENSE
    └── README.md

## Instructions:
1. Clone GIT repository 

    - git clone https://github.com/RikusFourie/Disaster-Response-Pipeline.git
2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3. Run the following command in the app's directory to run your web app. (Step 2 can be skipped if you wish to use the pretrained model)

    - `python run.py`

4. Go to http://127.0.0.1:3001/

## Homepage Preview
![HomePagePreview](https://user-images.githubusercontent.com/41228935/59554408-abb8ce00-8fa2-11e9-9612-a1090c190503.png)
## Classified Message Preview
![ClassifiedMessage](https://user-images.githubusercontent.com/41228935/59554427-e0c52080-8fa2-11e9-93e8-d563efe9ea94.png)
