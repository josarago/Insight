import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

df = pd.read_csv("winemag-data-130k-v2.csv")
disp_columns = ['Title','Description','Variety','Region','Country','Price ($)','Score (/100)']
df_columns = ['title','description','variety','region_1','country','price','points']

def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df_columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in df_columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[
    html.H4(children='Find some cool wine'),
    dcc.Input(id='input-text', value='initial value', type='text',style={'width': '100%','display': 'inline-block'}),
    html.Div(id='results'),
    # generate_table(df),

])

@app.callback(
    Output(component_id='results', component_property='children'),
    [Input(component_id='input-text', component_property='value')]
)
def do_something_cool(input_value):
    try:
        idx = int(input_value)
        return html.Table(
            # Header
            [html.Tr([html.Th(col) for col in disp_columns])] +
            # Body
            [html.Tr([html.Td(df.iloc[idx][col]) for col in df_columns])]
        )
    except Exception as e: print(e)



if __name__ == '__main__':
    app.run_server(debug=True)
