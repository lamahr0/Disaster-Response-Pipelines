import sys
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import re
import pickle
from multiprocessing.pool import ThreadPool as Pool


def load_data(database_filepath):
    """
        This function takes the filepath of the database and load the data from sqlite

        Arguments:
            database_filepath:file path of the database

        Returns:
            X: messages
            y: target features
            category_names: category names of the target features

     """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    # spilt the data into variables and targets
    df.drop('child_alone', axis='columns',inplace=True)
    X = df.iloc[:, 1]
    y = df.iloc[:, 4:]
    category_names = y.columns


    return X, y, category_names


def tokenize(text):
    """
        This function takes the messages normalize,tokenize,remove stop words and lemmatize them.

        Arguments:
            text:the messages.

        Returns: the cleaned tokens.

     """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # remove stop words
    words = [w for w in text if w not in stopwords.words("english")]
    # Reduce words to their root form
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
        This function builds a classifier using Random Forest and Multi Output Classifier
        then create a pipeline using count vectorizer, tf-idf and the classifier
        finally it uses grid search to find best parameters and return the best model

        Returns: the result of the grid search (best model)

    """
    # create Multi Output Classifier using random forest
    clf = MultiOutputClassifier(GradientBoostingClassifier())
    # create a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())]))
        ])),
        ('clf', clf)
    ])

    #parameters for gridsearch
    parameters = {
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__max_depth':[10]
    }

     #Create GridSearchCV
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
        This function evaluate the model using classifcation report

        Arguments:
            model: the classifier
            X_test:the test dataset for messages
            Y_test:the test dataset for target features
            category_names: category names of the target features

    """
    # predict on test dataset
    y_pred = model.predict(X_test)
    # Evaluate results
    i = 0
    for col in Y_test:
        print(classification_report(Y_test[col], y_pred[:, i]))
        


def save_model(model, model_filepath):
    """
        This function saves the model as pickle file

        Arguments:
            model:the classifier
            model_filepath:the filepath intended for the pickle file

    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
