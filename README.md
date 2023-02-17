# DisasterResponse
Analyze message data for disaster response

## File Description

## Installations

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
