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
wl.get_mean_region_vect_dict()


n_exact_max = 5
n_disp_max = 30
n_region_direct = 5
n_region_nlp = 10


DEFAULT_INPUT = ""
DEFAULT_REGION = 'All Regions'
REGION_OPTIONS = [{'label': DEFAULT_REGION, 'value': DEFAULT_REGION}]
REGION_OPTIONS.extend([{'label': x, 'value': x} for x in wl.mean_vects_dict.keys()])
nlp_style = { 'color': 'rgb(185, 25, 25)'}
default_style = { 'color': 'rgb(90, 90, 90)'}

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


def get_table(header_str,indexes,n_out=15,style=default_style):
    table_out = [html.Div(
                    style=style,
                    children = [
                        html.H2 (style={'margin-top': '1%'},children=header_str),
                        html.Table(
                            [html.Tr([html.Th(col) for col in DISP_COLUMNS])] +
                            [html.Tr([
                                html.Td(wl.df[wl.df.index==idx][col]) for col in DF_COLUMNS
                                ]) for idx in indexes[:n_out]]
                            )
                        ]
                    )
                ]
    return table_out

for css in external_css:
    app.css.append_css({"external_url": css})

app.layout = html.Div(children=[
                html.Div(className='searchdiv',
                        children=[
                            html.H1(children='SpeakEasy Wine'),
                            html.Div([
                                dcc.Dropdown(
                                    id='region-dropdown',
                                    options=REGION_OPTIONS,
                                    value=DEFAULT_REGION,
                                    style={
                                        # 'display': 'inline-block',
                                        'width': '65%',
                                        },
                                    clearable=True
                                    ),
                                dcc.Input(
                                        placeholder="Describe the wine you are looking for. Example: 'fresh everyday dry wine with notes of citrus'",
                                        id='wine-search-bar',
                                        value=DEFAULT_INPUT,
                                        type='text',
                                        style={
                                            'width': '100%',
                                            'margin-left':'1%',
                                            # 'display': 'inline-block',
                                            },
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

@app.callback(
    Output(component_id='results', component_property='children'),
    [Input(component_id='wine-search-bar', component_property='value'),
    Input(component_id='region-dropdown', component_property='value')
    ]
)

def display(input_value,region_name):
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
    desc = wl.tokenize(input_value,vocab=list(wl.model.wv.vocab.keys()))
    desc_cond = len(desc)>0
    joined_kw = " - ".join(desc)
    region_cond = not (region_name==DEFAULT_REGION or region_name==None)
    if not desc and not region_cond:
        kids.extend([html.P("This how SpeakEasy Wine works...")])
    elif not desc and region_cond:
        # direct regions
        all_regional_indexes = wl.df[wl.df.region_1==region_name].index
        regional_indexes, _ = wl.get_doc2vec_region_wines(
                                                        region_name,
                                                        include_indexes=all_regional_indexes,
                                                        topn=500,
                                                        method='mean',
                                                        desc=None
                                                    )
        direct_region_table = get_table(
                    "Here are typical wines from the region '{}'.".format(region_name),
                        regional_indexes,n_out=n_region_direct
                    )
        kids.extend(direct_region_table)
        # doc2vec regions:
        doc2vec_regional_indexes, _ = wl.get_doc2vec_region_wines(
                        region_name,
                        exclude_indexes=all_regional_indexes,
                        topn=10,
                        method='mean',
                        desc=None)
        doc2vec_region_table = get_table(
                    "We also found wines from different regions that taste similar.",
                    doc2vec_regional_indexes,
                    n_out=n_region_nlp,
                    style=nlp_style,
                )
        kids.extend(doc2vec_region_table)
    elif desc and region_cond:
        # direct regions:
        regional_with_kw = wl.get_direct_region_wines(region_name,desc)
        if len(regional_with_kw)>0:
            with_kw_table = get_table(
                    "Here are wines from the region '{}' matching the keywords '{}'.".format(
                                                                region_name,
                                                                joined_kw
                                                                ),
                    regional_with_kw,
                    n_out=n_region_direct
                )
            kids.extend(with_kw_table)
        else:
            sorry_message = "We didn't find any wine from the region '{}' matching the keyword(s) '{}', sorry.".format(
                                                            region_name,
                                                            joined_kw
                                                            ),
            kids.extend([html.H3(children=sorry_message)])
        all_regional_indexes = wl.df[wl.df.region_1==region_name].index
        doc2vec_regional_with_kw, _ = wl.get_doc2vec_region_wines(
                    region_name,
                    desc=desc,
                    exclude_indexes=all_regional_indexes,
                    weight=.6,
                    topn=500,
                    )
        # print(doc2vec_regional_with_kw)
        doc2vec_region_kw_msg = "We found wines that have characteristics of '{}' and '{}'.".format(region_name,joined_kw)
        doc2vec_region_kw_table = get_table(
                        doc2vec_region_kw_msg,
                        doc2vec_regional_with_kw,
                        n_out=n_region_direct,
                        style=nlp_style,
                    )
        kids.extend(doc2vec_region_kw_table)
    elif desc and not region_cond:
        exact_indexes = wl.get_exact_match_from_description(desc)
        kids.extend([html.Div(
            style=nlp_style,
            children = [
                    html.P(
                            style={'margin-top': '1%'},
                            children="Using the key words:"
                        ),
                    html.H4(children=joined_kw),
                ]
            )
        ])
        docs2vec_indexes = wl.get_doc2vec_wines_from_desc(desc,topn=20)
        docs2vec_final_indexes = [idx for idx in docs2vec_indexes if idx not in exact_indexes]
        exact_match_str = get_exact_match_str(len(exact_indexes),n_exact_max)
        if len(exact_indexes)>0:
            exact_match_kid = get_table(exact_match_str,exact_indexes,n_out=5)
        else:
            exact_match_kid = [
                html.P(
                        style={'margin-top': '1%'},
                        children=exact_match_str
                    ),
            ]
        kids.extend(exact_match_kid)
        # some text to tell user this is NLP
        nlp_table = get_table(
                "Our Machine Learning algorithm found other wines you might like",
                docs2vec_final_indexes,
                n_out=15)
        doc2vec_kid = [
                html.Div(
                    style=nlp_style,
                    children = nlp_table
                )
            ]
        kids.extend(doc2vec_kid)

    return html.Div(
        children=kids
        )

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', debug=True)
