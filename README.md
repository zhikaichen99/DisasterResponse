# DisasterResponse
Analyze message data for disaster response

## 

## Repository Structure and File Description

```markdown
├── app
│   ├── templates
│   │   ├── go.html                       # website page after a query has been made
│   │   └── master.html                   # website homepage
│   ├── app.py                            # python file to get the flask web app running
├── data
│   ├── disasterMessages.db               # distaster messages database
│   ├── disaster_categories.csv           # categories csv file
│   ├── disaster_messages.csv             # messages csv file
│   └── process_data.py                   # python script to process data for model training
├── models
│   └── train_classifier.py               # python script that builds, train, evaluate, and save model. 
├── ETL_Pipeline_Preparation.ipynb        # Data processing shown in a jupyter notebook
├── ML_Pipeline.ipynb                     # Model training shown in a jupyter notebook
├── README.md                             # Readme file            

```

## Installations

To run this project, the following libraries and packages must be installed:

* Pandas
* Natural Language Toolit (NLTK)
* SQLAlchemy
* Pickle
* Sys
* Scikit-learn
* Json
* Plotly
* Flask

## How to Interact with the Project

1. Clone the repository to your local machine using the following command:
```
git clone https://github.com/zhikaichen99/DisasterResponse.git
```
2. Run the `process_data.py` script to process the data for model training by running the following command:
```
python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disasterMessages.db
```
3. Run the `train_classifier.py` script to train a machine learning model and generate the model in the form of a pickle file. Run the following command:
```
python models/train_classifier.py data/disasterMessages.db models/model.pkl
```
4. Navigate to the `app` directory

5. Run the following command in the terminal top run the web app:
```
python app.py
```
