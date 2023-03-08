import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
        messages_filepath: Path to messages csv file
        categories_filepath: Path to categories csv file
    
    OUTPUT:
        df: Merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on = 'id')
    return df


def clean_data(df):
    """
    Input:
        df: Merged dataset
    
    Output:
        df: Cleaned dataset
    """
    #Create dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand = True)
    
    #Select the first row of the categories dataframe
    row = categories.iloc[0]
    
    #Use this row to extract a list of new column names for categories
    category_colnames = row.apply(lambda x:x[:-2])
    
    #Rename the columns of 'categories'
    categories.columns = category_colnames
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        #Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        #Convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    #Drop the original categories column from 'df'
    df = df.drop('categories', axis = 1)
    
    #Concatenate the original datafram with the new 'categories' dataframe
    df = pd.concat([df, categories], axis = 1)
    
    #Drop duplicates
    df.drop_duplicates(inplace = True)
    
    #Drop 'child_alone' column as it has only 0 (ZERO) values - as per our jupyter Notebook analysis
    df = df.drop('child_alone', axis = 1)
    
    #According to our Jupyter Notebook investigation, the maximum value for the'related' column is 2, which may be an error.
    df['related'] = df['related'].map(lambda x: 1 if x==2 else x)
        
    return df


def save_data(df, database_filename):
    """
    Input:
        df: Cleaned data
        database_filename: File extension (.db) used to identify the SQLite database
    
    Output:
        None: Save cleaned data into sqlite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('DisasterResponse_table', engine, index = False, if_exists = 'replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning dataset...')
        df = clean_data(df)
        
        print('Saving dataset...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned dataset saved to database!')
    
    else:
        print('Kindly supply the directory structures where the messages and their respective folders can be found. '\
              'use two datasets, one as the first argument and the other as the '\
              'as well as the location in the database where the sanitized data should be stored '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()