# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import warnings
from sklearn.exceptions import DataConversionWarning

nltk.download(['punkt', 'wordnet'])

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings('ignore')

def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('ClassifiedMessages', engine)
    X = df.message.values
    Y = df[df.columns[4:]].values
    cat_names=df.columns[4:]
    
    return X, Y, cat_names


def tokenize(text):
    #replacing urls
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    for i in urls:
        text = text.replace(i, "urlplaceholder")
        
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    #Tokenize words
    words = word_tokenize(text)
    
    #Lemmatize words
    lemmed = [WordNetLemmatizer().lemmatize(w.strip()) for w in words]
    
    return lemmed


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
        ('clf', MultiOutputClassifier(OneVsRestClassifier(LinearSVC(random_state = 0))))
    ])
    
    parameters = {
        'features__text_pipeline__vect__max_df': (0.5,0.75, 1.0),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__estimator__loss': ['hinge','squared_hinge'],
        'clf__estimator__estimator__multi_class': ['ovr', 'crammer_singer'],
        'clf__estimator__estimator__max_iter': [1000,2000,5000]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    X_pred = model.predict(X_test)
    
    eval_dict={'column':[],'f1_score':[],'precision':[],'recall':[]}
    
    for column in range(len(category_names)):
        f1s=f1_score(Y_test[:,column],X_pred[:,column], average='micro')
        prs=precision_score(Y_test[:,column],X_pred[:,column], average='micro')
        res=recall_score(Y_test[:,column],X_pred[:,column], average='micro')
        eval_dict['column'].append(category_names[column])
        eval_dict['f1_score'].append(f1s)
        eval_dict['precision'].append(prs)
        eval_dict['recall'].append(res)
        
    print(pd.DataFrame.from_dict(eval_dict))


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, "wb"))


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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()