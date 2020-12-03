import os

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import joblib


import base64
import datetime
import io


app = dash.Dash(__name__, title='Fraud Analytics')

server = app.server
app.config.suppress_callback_exceptions = True

pio.templates.default = "plotly_dark"

#load pickle file for prediction model
def loadmodel(filename):
    loaded_model = joblib.load(open(filename, 'rb'))
    return loaded_model
pipeline = loadmodel("notebooks/final_pickle/pickle_website_compress_2.pkl")

df = pd.read_csv("~/code/mariia-rum/project_fraud/notebooks/Gilles/val_data.csv")
print('done loading app')

#html.Link(rel="preconnect", href="https://fonts.gstatic.com")
#html.Link(href="https://fonts.googleapis.com/css2?family=Quicksand&display=swap", rel="stylesheet")

print('starting the app')

#fig = px.bar(df, x="probability_score", color='probability_score', color_continuous_scale = px.colors.diverging.RdYlGn)
#fig = px.histogram(df, x="probability_score")

app.layout = html.Div(children=[
            html.H4(children='Card Transactions'),
            html.H2(children='Fraud Analytics Overview'),
            html.Div(),
    #load files
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'color':'white',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=True,
    ),
    dcc.Loading(
            id="loading-1",
            type="default",
            children=html.Div(id="loading-output-1")
        ),
    html.Div(id='output-data-upload'),

     # dcc.Graph(
     #      id='example-graph',
     #      figure=fig),
    # N, bins, patches = plt.hist(X_val['proba'], alpha=0.5, bins=5, range=(0,1))
    #     for i in range(0,1):
    #         patches[i].set_facecolor('#A0E0B4')
    #     for i in range(1,2):
    #         patches[i].set_facecolor('#B3EFBC')
    #     for i in range(2,3):
    #         patches[i].set_facecolor('#FFDC72')
    #     for i in range(3,4):
    #         patches[i].set_facecolor('#FFA99B')
    #     for i in range(4,5):
    #         patches[i].set_facecolor('#F88484')


    # plt.xlabel('Probability of transaction being fraudulent')
    # plt.ylabel('Number of transactions')

    # plt.show;


    # data=[go.Histogram(x = x1, cumulative_enabled = True)]
    # fig = go.Figure(data)
    # iplot(fig)

    ])

print("Loading in data from data/")

#STEP2: USE PIPELINE:
def pipelines(pipeline, df):
    proba_score = pipeline.predict_proba(df)
    df['probability_score'] = proba_score[:,1]
    return df

#STEP 3: DROPPING COLOMNS:
def drop_cols(df):
     df = df.drop(columns=['card1',
        'card2', 'addr1', 'card5', 'D15', 'C13', 'D2', 'D10',
        'D4','dist_mean', 'dist_median', 'dist_mean_rel',
        'dist_median_rel'])
     return df

def rename(df):
    df.rename(columns={'TransactionAmt': 'Transaction Amount', },inplace=True)
    return df

def deencode_weekdays(weekday):
    if weekday == 0.0:
        return 'Monday'
    elif weekday == 1.0:
        return 'Tuesday'
    elif weekday == 2.0:
        return 'Wednesday'
    elif weekday == 3.0:
        return 'Thursday'
    elif weekday == 4.0:
        return 'Friday'
    elif weekday == 5.0:
        return 'Saturday'
    elif weekday == 6.0:
        return 'Sunday'


#STEP 1: GET THE DATA
def parse_contents(contents, filename, date):
    print('starting parse' )
    print(filename)
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
        else:
            return html.Div(['We only accept CSV Files'])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    try:
        df = pipelines(pipeline, df)
        df = drop_cols(df)
        df = rename(df)
        df["weekday"] = df["weekday"].apply(lambda x: deencode_weekdays(x))
    except Exception as e:
        print(e)
        return html.Div([
            'CSV is shit'
        ])

    return html.Div([
          html.H5(filename),
          html.H6(datetime.datetime.fromtimestamp(date)),
          dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
            style_header={'backgroundColor':'#4E4B5D', 'color': 'white', 'border': '1px solid black', 'textAlign':'center', 'fontFamily':'Open Sans'},
            style_cell={'minWidth': 120, 'maxWidth': 120, 'width': 120, 'padding':'5px'},
            fixed_rows={'headers': True},
            style_data_conditional=[
                            {'if':{'column_id' :'probability_score',
                                'filter_query' : '{probability_score} <= 0.2'
                            },
                            'backgroundColor':'#A0E0B4'},
                            {'if':{'column_id' :'probability_score',
                                'filter_query' : '0.2 < {probability_score} <= 0.4'
                            },
                             'backgroundColor':'#B3EFBC'},
                            {'if':{'column_id' :'probability_score',
                                'filter_query' : '0.4 < {probability_score} <= 0.6'
                            },
                             'backgroundColor':'#FFDC72'},

                            {'if':{'column_id' :'probability_score',
                                'filter_query' : '0.6 < {probability_score} <= 0.8'
                            },
                             'backgroundColor':'#FFA99B'},

                            {'if':{'column_id' :'probability_score',
                                'filter_query' : '{probability_score} > 0.8'
                            },
                             'backgroundColor':'#F88484'}

                            ],

    style_table={'height': '1000px', 'whiteSpace': 'normal', 'padding': '2px'
        },
    style_data={'border':'1px solid #c6d4fb', 'fontFamily':'sans-serif'}
        ),
        html.Hr(),  # horizontal line
        # For debugging, display the raw contents provided by the web browser
         html.Div('Raw Content'),
         html.Pre(contents[0:200] + '...', style={
             'whiteSpace': 'pre-wrap',
             'wordBreak': 'break-all'
         })
    ])

#fig.update_layout(template=plotly_dark)

#style_todo = {"display": "inline", "margin": "10px"}
#style_done = {"textDecoration": "line-through", "color": "#888"}
#style_done.update(style_todo)

@app.callback(Output('output-data-upload', 'children'),
              Input('upload-data', 'contents'),
              State('upload-data', 'filename'),
              State('upload-data', 'last_modified'))
def update_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        children = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        return children

@app.callback(Output("loading-output-1", "children"), Input("loading-input-1", "value"))
def input_triggers_spinner(value):
    time.sleep(1)
    return value

print("hola")
if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)

