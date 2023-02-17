import sys
import pandas as pd

from sqlalchemy import create_engine

def load_data(messages_file_path, categories_file_path):
    """
    Loads messages dataset and categories dataset. Merges the two datasets on the id column

    Input:
        messages_file_path: file path to the messages csv file
        categories_file_path: file path to the categories csv file

    Output:
        df: merged dataframe
    
    """
    # Load  messages dataset
    messages = pd.read_csv(messages_file_path)
    # Load categories dataset
    categories = pd.read_csv(categories_file_path)
    # Merge datasets using the id column
    df = pd.merge(messages, categories, on = 'id')
    # Return merged dataset
    return df

def clean_data(df):
    """
    Function that cleans the data

    Input: 
        df: merged dataframe

    Output:
        df: cleaned dataframe
    """
    # Create a dataframe of the individual category columns
    categories = df['categories'].str.split(';', expand = True)
    # Use the first row to extract a list of new column names for categories
    row = categories.iloc[0]
    # Select everything up to the second last character of string
    category_column_names = row.apply(lambda x: x[:-2])
    # Rename the columns of the categories dataframe
    categories.columns = category_column_names

    # Convert category values to 0 and 1. A 0 inidicating the message does not belong to that category and 1 indicating it does belong to that category
    for column in categories:
        # set each value to be the last character of the string (which is the number). Replaces value with 0 if it is not 0 or 1
        categories[column] = categories[column].apply(lambda x: '0' if not x[-1].isdigit() or int(x[-1]) not in [0, 1] else x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # Drop the original categories column from df
    df.drop('categories', axis = 1, inplace = True)
    # Concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis = 1)
    # Drop duplicates
    df.drop_duplicates(inplace = True)
    # Return cleaned df
    return df


def save_data(df, database_filename):
    """
    Stores df in an SQLite database

    Input:
        df: dataframe that is to be stored in the database
        database_filename: name of the database
    
    Output:
        None
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterMessagesTable', engine, index = False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()