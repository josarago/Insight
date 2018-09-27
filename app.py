import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
from wine_and_cheese_utils import df_columns, WineList
from gensim.models.doc2vec import Doc2Vec

# df = pd.read_csv("winemag-data-130k-v2.csv")
wl = WineList(file='cleaned')
disp_columns = ['Title','Description','Variety','Region','Country','Price ($)','Score (/100)']
model = Doc2Vec.load('doc2vec_on_region_1_no_region_variety_full_dataset.model')

# def generate_table(dataframe, max_rows=10):
#     return html.Table(
#         # Header
#         [html.Tr([html.Th(col) for col in df_columns])] +
#
#         # Body
#         [html.Tr([
#             html.Td(dataframe.iloc[i][col]) for col in df_columns
#         ]) for i in range(min(len(dataframe), max_rows))]
#     )


app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[
    html.H4(children='Cooler Wines'),
    dcc.Slider(
        id='my-slider',
        min=5,
        max=100,
        step=1,
        value=10,
    ),
    html.Div(id='slider-output-container'),
    dcc.Input(
            id='input-text',
            value='grapefruit mineral dry summer',
            type='text',
            style={'width': '100%','display': 'inline-block','fontColor': 'blue'},
        ),
    html.Div(id='results'),
    # generate_table(df),

])

@app.callback(
    dash.dependencies.Output('slider-output-container', 'children'),
    [dash.dependencies.Input('my-slider', 'value')])
def update_output(value):
    return 'Max Price $"{}"'.format(value)

@app.callback(
    Output(component_id='results', component_property='children'),
    [Input(component_id='input-text', component_property='value')]
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
    # tokenize the input_value
    
    # for each token
        # check if varieties or regions are present

    # if there is more than one variety or one region
        #throw some error

    # if there is only one variety
        # return message "so yeah we have n wines of this variety but that's boring. try .. instead"
    # if there is only one region
        # return message "so yeah we have n wines of this region but that's boring. try .. instead"

    # if user enter keywords
        # try to find exact results
            # if some exact matches are found display them

        # then display ML suggested options

    try:
        indexes, new_desc = wl.get_wines_from_desc(input_value,model,topn=50)
        # print(indexes)
        return html.Div(
            children=[
                    html.P(",".join(new_desc)),
                    html.Table(
                        # Header
                        [html.Tr([html.Th(col) for col in disp_columns])] +
                        # Body
                        [html.Tr([
                            html.Td(wl.df[wl.df.index==idx][col]) for col in df_columns
                        ]) for idx in indexes]
                    ),

                ]
            )

    except Exception as e:
        print(e)
        return "enter a integer"


if __name__ == '__main__':
    app.run_server(debug=True)
