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
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

#load pickle file for prediction model
def loadmodel(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

#Style of the app
app = dash.Dash(__name__, title='Fraud Analytics')


server = app.server


print("Loading in data from data/")


df = pd.read_csv("~/code/mariia-rum/project_fraud/notebooks/Gilles/val_data.csv")
pipeline = loadmodel("notebooks/Gilles/pipeline_8.pkl")

def pipelines(pipeline, df):
    proba_score = pipeline.predict_proba(df)
    df['probability_score'] = proba_score[:,1]
    return df.round(2)


# def generate_table(dataframe, max_rows=10):
#     return html.Table([
#         html.Thead(
#             html.Tr([html.Th(col) for col in dataframe.columns])
#         ),
#         html.Tbody([
#             html.Tr([
#                 html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
#             ]) for i in range(min(len(dataframe), max_rows))
#         ])
#     ])

df = pipelines(pipeline, df)

def drop_cols(df):
    df = df.drop(columns=['P_emaildomain_suffix', 'P_emaildomain_bin', 'card1',
       'card2', 'addr1', 'card5', 'D15', 'C13', 'D2', 'D10',
       'D4','dist_mean', 'dist_median', 'dist_mean_rel',
       'dist_median_rel'])
    return df

df = drop_cols(df)

# def colour_grade(df):
#     score = df.probability_score
#     if score.any() <= 0.2:
#         return "green",
#     elif score.any() <= 0.4:
#         return "lightgreen",
#     elif score.any() <= 0.6:
#         return "orange",
#     elif score.any() <= 0.8:
#         return "red",
#     else:
#         return "darkred"

def parse_contents(contents, filename, date):
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
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': i, 'id': i} for i in df.columns]
        ),

        html.Hr(),  # horizontal line

        # For debugging, display the raw contents provided by the web browser
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])


fig = px.bar(df, x="probability_score", color='probability_score', color_continuous_scale = px.colors.diverging.RdYlGn)
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
        multiple=True
    ),
    html.Div(id='output-data-upload'),


    dcc.Graph(
        id='example-graph',
        figure=fig),

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

    style_table={'height': '1000px', 'whiteSpace': 'normal'
        },
    style_data={'border':'1px solid #c6d4fb', 'fontFamily':'sans-serif'}
)])



# app.layout = html.Div([
#     html.Div('...'),
#     dcc.Input(id="new-item"),
#     html.Button("Add", id="add"),
#     html.Button("Clear Done", id="clear-done"),
#     html.Div(id="list-container"),
#     html.Div(id="totals")
# ])

style_todo = {"display": "inline", "margin": "10px"}
style_done = {"textDecoration": "line-through", "color": "#888"}
style_done.update(style_todo)

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

# @app.callback(
#     [
#         Output("list-container", "children"),
#         Output("new-item", "value")
#     ],
#     [
#         Input("add", "n_clicks"),
#         Input("new-item", "n_submit"),
#         Input("clear-done", "n_clicks")
#     ],
#     [
#         State("new-item", "value"),
#         State({"index": ALL}, "children"),
#         State({"index": ALL, "type": "done"}, "value")
#     ]
# )
# def edit_list(add, add2, clear, new_item, items, items_done):
#     triggered = [t["prop_id"] for t in dash.callback_context.triggered]
#     adding = len([1 for i in triggered if i in ("add.n_clicks", "new-item.n_submit")])
#     clearing = len([1 for i in triggered if i == "clear-done.n_clicks"])
#     new_spec = [
#         (text, done) for text, done in zip(items, items_done)
#         if not (clearing and done)
#     ]
#     if adding:
#         new_spec.append((new_item, []))
#     new_list = [
#         html.Div([
#             dcc.Checklist(
#                 id={"index": i, "type": "done"},
#                 options=[{"label": "", "value": "done"}],
#                 value=done,
#                 style={"display": "inline"},
#                 labelStyle={"display": "inline"}
#             ),
#             html.Div(text, id={"index": i}, style=style_done if done else style_todo)
#         ], style={"clear": "both"})
#         for i, (text, done) in enumerate(new_spec)
#     ]
#     return [new_list, "" if adding else new_item]


# @app.callback(
#     Output({"index": MATCH}, "style"),
#     Input({"index": MATCH, "type": "done"}, "value")
# )
# def mark_done(done):
#     return style_done if done else style_todo


# @app.callback(
#     Output("totals", "children"),
#     Input({"index": ALL, "type": "done"}, "value")
# )
# def show_totals(done):
#     count_all = len(done)
#     count_done = len([d for d in done if d])
#     result = "{} of {} items completed".format(count_done, count_all)
#     if count_all:
#         result += "- {}%".format(int(100 * count_done / count_all))
#     return result
# # app.layout = html.Div([
# #     html.H2('Hello World'),
# #     dcc.Dropdown(
# #         id='dropdown',
# #         # options=dropdown_dict,
# #         options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
# #         value='LA'
# #     ),
# #     html.Div(id='display-value')
# # ])

# # assume you have a "long-form" data frame
# # see https://plotly.com/python/px-arguments/ for more options


# @app.callback(dash.dependencies.Output('display-value', 'children'),
#               [dash.dependencies.Input('dropdown', 'value')])
# def display_value(value):
#     return 'You have selected "{}"'.format(value)


# <canvas class=
# "background"
# ></canvas>

# <script src=
# "path/to/particles.min.js"
# ></script>

if __name__ == '__main__':
    app.run_server(debug=True)
