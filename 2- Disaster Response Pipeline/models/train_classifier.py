import sys
import joblib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import re
from sqlalchemy import create_engine
import pickle
from scipy.stats import gmean
# import relevant functions/modules from the sklearn
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.base import BaseEstimator,TransformerMixin

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])

def load_data(database_filepath):
    """
    Load the Dataset from SQLite database
    
    Input:
         database_filepath:
            - type: str
            - Filepath for db file containing the dataset
    Output:
        - X: 
            - type: dataframe
            - Contains the features
        - y: 
            - type: dataframe
            - contains the prediction values
        - category_names:
            - type: list
            - contains the categories to be predicted
    """
    # Extract the dataset
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("message_table", engine)

    # split the dataset into features and predictions
    X = df['message']
    y = df.iloc[:,4:]

    # get the category names
    category_names = y.columns.values

    return X, y, category_names


def tokenize(text):
    """ 
    Normalize and Tokenize the text 

    Input:
        - text:
            - type: str
            - The string to be tokenized
    
    Output:
        - cleaned_tokens
            - type: list
            - The tokens extracted from the original string
    """

    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # removing stopwords
    tokens = [w for w in tokens if w not in stopwords.words("english")]
    # lemmatize and remove stop words
    lemmatizer = nltk.WordNetLemmatizer()
    # tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens


def build_model():
    """ 
    Pipeline Construction

    Output:
        - pipeline:
            - type: sklearn pipeline
            - pipeline containing the estimators and predictor
    """

    # Build the pipeline with the best parameters found via GridSearchCV
    pipeline = Pipeline(steps=[
        ('count_vectorizer', CountVectorizer(tokenizer=tokenize, min_df= 1)),
        ('tfidf_transformer', TfidfTransformer(use_idf= False)),
        ('clf', MultiOutputClassifier(RandomForestClassifier(min_samples_split=5, n_estimators= 10, verbose=True)))
    ])
    
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model

    Input:
        - model
            - type: sklearn pipeline
            - The fitted pipeline
        - X_test
            - type: df
            - The features for testing
        - y_test
            - type: df
            - The ground truth predictions
        - category_names
            - type: list
            - The category names
    """
    
    # get the predictions from the fitted model
    y_pred_test = model.predict(X_test)

    accuracy_lst_pipeline1 = []
    # Calculate the accuracy for each of them.
    for i in range(len(category_names)):
        print(f'Category {i}: {category_names[i]} ')
        print(classification_report(Y_test.iloc[:, i].values, y_pred_test[:, i]))
        acc = accuracy_score(Y_test.iloc[:, i].values, y_pred_test[:, i])
        print(f'Accuracy {acc}\n\n')
        accuracy_lst_pipeline1.append(acc)


def save_model(model, model_filepath):
    """
    Saves the model as a pickle file

    Input:
        - model
            - type: sklearn fitted model
        - model_filepath:
            - type: str
            - The path where the model should be saved 
    """
    joblib.dump(model, model_filepath)


def main():
    """
    Main Function to run the ETL pipeline
    """
    
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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()