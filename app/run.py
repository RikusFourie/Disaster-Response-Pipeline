import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Function Description:
        Tokanizer for text
    
    Input:
        text: Text messages
        
    Output:
        Tokenized words (List)
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('ClassifiedMessages', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Function Description:
        Funtion to display homepage and display graphs
    
    Input:
        None
        
    Output:
        Rendered HTML
    """
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message'].reset_index().sort_values(by='message', ascending=False)
    
    # Adding Catagory Count Graph
    def countvals(filtered_df):
        CatDF=filtered_df.loc[:,filtered_df.columns.isin(list(filtered_df.columns[4:]))]

        countdict={'Catagory':[],'Value':[]}
        for i in CatDF.columns:
            countdict['Catagory'].append(i)
            countdict['Value'].append(sum(CatDF[i].values))

        return pd.DataFrame.from_dict(countdict).sort_values(by='Value', ascending=False)

    countdf=countvals(df)
    
    # Adding Catagory Count Graph per Genre
    countdfnews=countvals(df.loc[df.genre=='news'])[:10]
    countdfdirect=countvals(df.loc[df.genre=='direct'])[:10]
    countdfsocial=countvals(df.loc[df.genre=='social'])[:10]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_counts.genre,
                    y=genre_counts.message
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=countdf.Catagory,
                    y=countdf.Value
                )
            ],

            'layout': {
                'title': 'Distribution of Message Catagory',
                "height": 500,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "<br><br><br> Catagory",
                    'tickangle':45
                },
                "margin": {
                  "b": 150
                }
            }
        },
        {
            'data': [
                Bar(
                    x=countdfnews.Catagory,
                    y=countdfnews.Value
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Message Catagory Where Genre Is News',
                "height": 500,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "<br><br><br> Catagory",
                    'tickangle':45
                },
                "margin": {
                  "b": 150
                }
            }
        },
        {
            'data': [
                Bar(
                    x=countdfdirect.Catagory,
                    y=countdfdirect.Value
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Message Catagory Where Genre Is Direct',
                "height": 500,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "<br><br><br> Catagory",
                    'tickangle':45
                },
                "margin": {
                  "b": 150
                }
            }
        },
        {
            'data': [
                Bar(
                    x=countdfsocial.Catagory,
                    y=countdfsocial.Value
                )
            ],

            'layout': {
                'title': 'Top 10 Distribution of Message Catagory Where Genre Is Social',
                "height": 500,
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "<br><br><br> Catagory",
                    'tickangle':45
                },
                "margin": {
                  "b": 150
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Function Description:
        Classify user input into catagories.
    
    Input:
        None
        
    Output:
        Rendered HTML
    """
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='127.0.0.1', port=3001, debug=True)


if __name__ == '__main__':
    main()