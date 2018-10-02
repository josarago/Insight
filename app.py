import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from wine_and_cheese_utils import df_columns, WineList

# df = pd.read_csv("winemag-data-130k-v2.csv")
disp_columns = [
            'Title                          ',
            'Description',
            'Variety',
            'Region',
            'Country',
            'Price($)',
            'Score(/100)',
        ]

wl = WineList(file='cleaned')
# create and store the TaggedDocument list
wl.get_tagged_data()
# import (or retrain) the Doc2Vec model
wl.get_doc2vec_model()
n_disp_max = 30

external_css = [
        "//fonts.googleapis.com/css?family=Pacifico:400,300,600",
        "//fonts.googleapis.com/css?family=Comfortaa:400,300,600",
    ]

app = dash.Dash(__name__)
app.title = "SpeakEasy Wine"
# app.config['suppress_callback_exceptions']=True

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(children=[
                html.Div(className='searchdiv',
                        children=[
                            html.H1(children='SpeakEasy Wine'),
                            dcc.Input(
                                    placeholder='Describe the wine you are looking for',
                                    id='wine-search-bar',
                                    value='',
                                    type='text',
                                    style={
                                            'width': '96%',
                                            'margin-left': '2%',
                                            'margin-right': '2%',
                                            'margin-bottom': '1%',
                                            # 'display': 'inline-block',
                                        },
                                    ),
                            ]
                        ),
                html.Div(id='results'),
                ]
            )
#
# @app.callback(
#     dash.dependencies.Output('slider-output-container', 'children'),
#     [dash.dependencies.Input('price-range-slider', 'value')])
# def update_output(value):
#     return 'Price range: $ {} - {}'.format(value[0],value[1])

@app.callback(
    Output(component_id='results', component_property='children'),
    [Input(component_id='wine-search-bar', component_property='value')]
)

def display(input_value):
    """
        behavior:
            user can enter:
                - variety (will be shown wine of that variety)
                - region (will be shown wine of that variety)
                - any keywords

            always start by presenting results that have exact matches,
            then present results from Doc2vec
    """
    kids = []
    if input_value=="":
        kids.extend([html.P("Describe the wine you are looking for")])
    else:
        exact_indexes, desc = wl.get_exact_match_from_description(input_value)
        docs2vec_indexes = wl.get_doc2vec_wines_from_desc(desc,topn=20)
        docs2vec_final_indexes = [idx for idx in docs2vec_indexes if idx not in exact_indexes]
        if len(exact_indexes)>0:
            exact_match_kid =[
                html.H3("We found {} exact matches:".format(len(exact_indexes))),
                # Header Table
                html.Table(
                    [html.Tr([html.Th(col) for col in disp_columns])] +
                    [html.Tr([
                        html.Td(wl.df[wl.df.index==idx][col]) for col in df_columns
                    ]) for idx in exact_indexes[:5]]
                    ),
            ]
        else:
            exact_match_kid = [
                html.P(
                        style={'margin-top': '1%'},
                        children="We didn't find any exact match BUT!"
                    ),
            ]
        kids.extend(exact_match_kid)
        # some text to tell user this is NLP
        doc2vec_kid = [
                html.Div(
                    style={ 'color': 'rgb(185, 25, 25)'},
                    children = [
                    html.P(
                            style={'margin-top': '1%'},
                            children="Using the key words:"
                        ),
                    html.H4(format(" - ".join(desc))),
                    html.P("we found wines you might like"),
                    html.Table(
                        [html.Tr([html.Th(col) for col in disp_columns])] +
                        [html.Tr([
                            html.Td(wl.df[wl.df.index==idx][col]) for col in df_columns
                        ]) for idx in docs2vec_final_indexes[:15]]
                    ),
                    ]
                )
            ]
        kids.extend(doc2vec_kid)

        return html.Div(
            children=kids
            )

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
