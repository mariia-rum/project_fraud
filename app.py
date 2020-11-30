import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
#from project_fraud.lib import drop_many_missing_values
import matplotlib.pyplot as plt

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

server = app.server

#data = drop_many_missing_values()


#vals = plt.hist(data['TransactionDT'] / (3600*24), bins=1800)
#plt.xlim(70, 78)
#plt.xlabel('Days')
#plt.ylabel('Number of transactions')
#plt.ylim(0,1000)

app.layout = html.Div([
    html.H2('Hello World'),
    dcc.Dropdown(
        id='dropdown',
        options=[{'label': i, 'value': i} for i in ['LA', 'NYC', 'MTL']],
        value='LA'
    ),
    html.Div(id='display-value')
])

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options


@app.callback(dash.dependencies.Output('display-value', 'children'),
              [dash.dependencies.Input('dropdown', 'value')])
def display_value(value):
    return 'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)
