import sys
import re
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    #Import Data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #Merge Datasets
    df = pd.merge(messages, categories,on='id')

    return df


def clean_data(df):
    #Split categories into separate category columns
    categories = df.categories.str.split(";", n=36,expand=True)
    
    row = categories.iloc[0]
    get_names=lambda x: re.split('-',x)[0]
    category_colnames = row.apply(get_names).values
    
    categories.columns = category_colnames
    
    get_col_names=lambda x: x[-1]
    
    #Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(get_col_names)

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
        #Replace categories column in df with new category columns
    df = df.drop(['categories'], axis=1)
    df = pd.concat([df, categories], axis = 1)
    df=df.drop_duplicates(subset=df.columns.difference(['id']))
    return df


def save_data(df, database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('ClassifiedMessages', engine, index=False)


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