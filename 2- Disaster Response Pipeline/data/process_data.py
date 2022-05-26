import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load & Merge 'messages' & 'categories' datasets
    
    inputs:
        - messages_filepath: 
            - type: str 
            - Filepath for csv file containing messages dataset.
        - categories_filepath: 
            - type: str 
            - Filepath for csv file containing categories dataset.
       
    outputs:
        - df: 
            - type: dataframe 
            - Dataframe containing the merged content of messages & categories datasets.
    """

    # read the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the 2 datasets on the "id" column
    df = pd.merge(messages, categories, on="id")

    return df


def clean_data(df):
    """
    Clean the dataframe by removing duplicates & converting categories from strings to binary values.
    
    Inputs:
        - df: 
            - type: dataframe 
            - Dataframe containing merged content of messages & categories datasets.
       
    Outputs:
        - df: 
            - type: dataframe 
            - Dataframe containing cleaned version of input dataframe.
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0, ]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x.split("-")[0])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(
            lambda x: x.split("-")[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')

    #  drop the original categories column from `df`
    df.drop('categories', inplace=True, axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates(subset=['message'])
    
    # drop child alone column since it consists of all 0
    new_df = df.drop('child_alone', axis=1)
    
    # change value 2 to 1 in related column
    new_df['related']=new_df['related'].apply(lambda x: 1 if x == 2 else x)
    
    # One Hot Encode the genres
    ohe_df = pd.get_dummies(new_df[['genre']], drop_first=True)
    
    # Concatinate the final dataframe
    final_df = pd.concat([new_df,ohe_df], axis=1, sort=False)    

    return final_df

def save_data(df, database_filename):
    """
    Save the Dataframe into  SQLite database.
    
    Inputs:
        - df: 
            - type: dataframe 
            - Dataframe containing cleaned version of merged message and categories data.
        - database_filename: 
            - type: str 
            - Filename for output database.
    """
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_table', engine, index=False, if_exists= 'replace')


def main():
    """
    Main Function to run the ETL pipeline
    """

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