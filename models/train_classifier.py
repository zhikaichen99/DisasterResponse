import sys

import pandas as pd
import nltk
import pickle

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report


from sklearn.model_selection import GridSearchCV



from sqlalchemy import create_engine

nltk.download(['punkt', 'wordnet'])

def load_data(database_filename):
    """
    Load the data from the SQLite database and return the features of the data aswell as the labels

    Inputs:
        database_filename: filename of the database
    Outputs:
        X: Features from the data
        Y: Labels of the data
    """
    # Load data from the database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df = pd.read_sql_table('DisasterMessagesTable', engine)
    # Features from the data
    X = df['message']
    # Labels from the data
    Y = df.iloc[:,4:]
    return X,Y


def tokenize(text):
    """
    Tokenize a string of text into individual words

    Inputs:
        text: The string that needs to be tokenized
    Outputs:
        clean_tokens: tokens
    """
    # Convert text to lowercase
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).strip()
        clean_tokens.append(clean_token)
    
    return clean_tokens


def build_model():
    """
    Build the machine learning pipeline

    Inputs:
        None
    Outputs:
        cv: Machine learning model classifier
    """
    # Build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Set up parameter grid
    parameters = {
        'clf__estimator__n_estimators': [50,100]
    }
    # Perform grid search 
    cv = GridSearchCV(pipeline, param_grid = parameters)
    # Return model
    return cv


def evaluate_model(model, X_test, Y_test):
    """
    Evaluate the performance of the machine learning model. Generate classification report

    Inputs:
        model: Machine learning classifier
        X_test: Features of the test dataset. Will be used to generate predictions
        Y_test: Labels of the test dataset. True values. Will be used to compare predictions versus true values.
    Outputs:
        None
    """
    # Generate predictions using trained model
    y_pred = model.predict(X_test)

    # Generate and print out classification reports for each category
    for idx, column in enumerate(Y_test):
        print('Classification report for Category: {}'.format(column))
        print(classification_report(Y_test[column], y_pred[:, idx]))


def save_model(model, model_filepath):
    """
    Save the trained machine learning model into a pickle file

    Inputs:
        model: trained machine learning model
        model_filepath: filepath of pickle file
    Outputs:
        None
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()