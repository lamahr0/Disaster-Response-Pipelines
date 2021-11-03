# Disaster-Response-Pipelines
</p>Machine learning approach to classify disaster messages into different categories, this is done to satisfy the requirements for the second project in the Udacity Data Scientist Nanodegree.</p>

### Table of Contents
- [Motivation](#motivation)
- [Dataset](#dataset)
- [File Descriptions](#file-descriptions)
- [Running The Code](#running-the-code)
- [Acknowledgements](#acknowledgements)

## Motivation 
In order to help emergency workers to work efficiently, this project main aim is to classify messages sent during a disaster so an emergency worker can identify what the message is about to send it to an appropriate disaster relief agency then the sender can get the help he needs whether its's about water, medical help, food, shelter .. etc.

## Dataset
the dataset is acquired through Figure Eight which is a machine learning and artificial intelligence company. the dataset contains real messages that were sent during disaster events.

#### Features:
id: unique identifier.<br>
message: the text message.<br>
original: the message in it's original language.<br>
genre: the source of the message.<br>
categories: which category the message belongs to.<br>


## File Descriptions
the repository contains three folders:
1. App
  * run.py: the web app run function
  * templates
    * go.html
    * master.html: home html page 
2. Data
  * process_data.py: python script for ETL pipeline
  * DisasterResponse.db: sqlite database for storing the cleaned dataset
  * ETL Pipeline Preparation.ipynb: Jupyter notebook for ETL pipeline
  * disaster_messages.csv: first dataset file containing messages
  * disaster_categories.csv: second dataset file containing categories 
3. Models
  * ML Pipeline Preparation.ipynb: Jupyter notebook for machine learning pipeline
  * train_classifier.py: python script for machine learning pipeline
  * classifier.pkl: model saved as a pickle file 

## Running The Code
* To run the ETL pipeline:
```python
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
```
* To run the Machine learning pipeline:
```python
python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
```
* To run the web app:
```python
python app/run.py
```

## Acknowledgements
- Udacity. <br>
- Figure Eight Inc. <br> 


