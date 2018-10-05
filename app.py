import pickle
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from wine_utils import DF_COLUMNS, WineList

# df = pd.read_csv("winemag-data-130k-v2.csv")
DISP_COLUMNS = [
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
# wl.get_tagged_data(,file_name = "tagged_data_set.pkl")
wl.get_tagged_data()#file_name = "tagged_data_set_regions_varieties_removed.pkl") #"")  #
# import (or retrain) the Doc2Vec model
wl.get_doc2vec_model()#from_file="doc2vec_regions_varieties_removed.model") #  f"doc2vec_regions_varieties_removed.model")
n_exact_max = 5
n_disp_max = 30

with open ("mean_region_docvecs_dict.pkl", 'rb') as fp:
    MEAN_VECT_DICT =  pickle.load(fp)

REGION_OPTIONS = [{'label': 'All Regions', 'value': 'All Regions'}]
REGION_OPTIONS.extend([{'label': x, 'value': x} for x in MEAN_VECT_DICT.keys()])

external_css = [
        "//fonts.googleapis.com/css?family=Pacifico:400,300,600",
        "//fonts.googleapis.com/css?family=Comfortaa:400,300,600",
    ]

app = dash.Dash(__name__)
app.title = "SpeakEasy Wine"

def get_exact_match_str(n, n_exact_max):
    if n==0:
        return "We couldn't find any exact match"
    elif n==1:
        return "We found 1 exact match"
    else:
        return "We found {} exact matches. Here are {} of them".format(n,n_exact_max)
# app.config['suppress_callback_exceptions']=True

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(children=[
                html.Div(className='searchdiv',
                        children=[
                            html.H1(children='SpeakEasy Wine'),
                            html.Div([
                                dcc.Dropdown(
                                    options=REGION_OPTIONS,
                                    value="All Regions",
                                    style={
                                        # 'display': 'inline-block',
                                        'width': '65%',
                                        },
                                    clearable=True
                                    ),
                                dcc.Input(
                                        placeholder="Describe the wine you are looking for. Example: 'fresh everyday dry wine with notes of citrus'",
                                        id='wine-search-bar',
                                        value='',
                                        type='text',
                                        # style={
                                        #     'width': '100%',
                                        #     # 'display': 'inline-block',
                                        #     },
                                        ),
                                    ],
                                    style={
                                        'display': 'flex',
                                        'margin-left':'2%',
                                        'margin-right':'2%',
                                        },
                                )
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
        case 1:
        ------
        - 'All regions'
        - empty text input
        ------>     returns how-to use SpeakEasy Wine

        case 2:
        ------
        - 'Some Region'
        - empty text input
        ------>     returns wines in that regions + wines similar to wine from that regions
                    (top n most similar wines)

        case 3:
        ------
        - 'All regions'
        -  key words
        ------>     returns exact match for keywords + NLP suggestions

        case 4:
        ------
        - 'Some Region'
        -  key words
        ------>     returns exact matches + average docvec for the region with added keywordss

    """
    kids = []
    if input_value=="":
        kids.extend([html.P("Describe the wine you are looking for")])
    else:
        desc = wl.tokenize(input_value,vocab=list(wl.model.wv.vocab.keys()))
        print(" - ".join(desc))
        exact_indexes = wl.get_exact_match_from_description(desc)
        kids.extend([html.Div(
            style={ 'color': 'rgb(185, 25, 25)'},
            children = [
                    html.P(
                            style={'margin-top': '1%'},
                            children="Using the key words:"
                        ),
                    html.H4(" - ".join(desc)),
                ]
            )
        ])
        docs2vec_indexes = wl.get_doc2vec_wines_from_desc(desc,topn=20)
        docs2vec_final_indexes = [idx for idx in docs2vec_indexes if idx not in exact_indexes]
        exact_match_str = get_exact_match_str(len(exact_indexes),n_exact_max)
        if len(exact_indexes)>0:
            exact_match_kid =[
                html.H3(exact_match_str),
                # Header Table
                html.Table(
                    [html.Tr([html.Th(col) for col in DISP_COLUMNS])] +
                    [html.Tr([
                        html.Td(wl.df[wl.df.index==idx][col]) for col in DF_COLUMNS
                    ]) for idx in exact_indexes[:n_exact_max]]
                    ),
            ]
        else:
            exact_match_kid = [
                html.P(
                        style={'margin-top': '1%'},
                        children=exact_match_str
                    ),
            ]
        kids.extend(exact_match_kid)
        # some text to tell user this is NLP
        doc2vec_kid = [
                html.Div(
                    style={ 'color': 'rgb(185, 25, 25)'},
                    children = [
                    html.H3 (style={'margin-top': '1%'},children="Our Machine Learning algorithm found other wines you might like"),
                    html.Table(
                        [html.Tr([html.Th(col) for col in DISP_COLUMNS])] +
                        [html.Tr([
                            html.Td(wl.df[wl.df.index==idx][col]) for col in DF_COLUMNS
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
