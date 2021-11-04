#imports 
import sys
import pandas as pd
from sqlalchemy import create_engine 

def load_data(messages_filepath, categories_filepath):
    """
    This function loads the two datafiles and merge it into one dataframe

    Arguments:
        messages_filepath: the filepath of the messages file.
        categories_filepath: the filepath of the categories file.

    Returns:
        It returns a dataframe which is the merge of the two datafiles.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(messages,categories,how='outer' , on = 'id')
    
    #return the merged dataframe
    return df


def clean_data(df):
    """
    This function cleans the data:remove duplicates, null values 
    rename columns and split categories into individual columns 

    Arguments:
        df: merged data frame 

    Returns:
        returns the cleaned dataset 
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    #extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    # rename the columns of `categories`
    categories.columns = category_colnames
    #Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]
    # convert all coulmns datatype to integer
        categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories', axis='columns',inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df , categories], axis = 1)
    # drop duplicates
    df = df.drop_duplicates()
    
    
    return df 



def save_data(df, database_filename):
    """
    This function saves the dataframe into a sqlite database 

    Arguments:
        df: cleaned dataframe 
        database_filename: database name

    Returns:
        This is a description of what is returned.
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False)


def main():
    """
    The function is the main function it will do three things
    1-load two data files
    2-clean data
    3-save cleaned data into the database 

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