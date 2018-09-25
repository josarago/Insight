import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

df = pd.read_csv("winemag-data-130k-v2.csv")


def generate_table(dataframe, max_rows=10,columns = ['title','description','variety','region_1','country','price','points']):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )


app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[
    html.H4(children='Find some cool wine'),
    dcc.Input(id='my-id', value='initial value', type='text',style={'width': '100%','display': 'inline-block'}),
    html.Div(id='my-div'),
    generate_table(df),

])

@app.callback(
    Output(component_id='my-div', component_property='children'),
    [Input(component_id='my-id', component_property='value')]
)
def update_output_div(input_value):
    return 'You\'ve entered "{}"'.format(input_value.lower())


if __name__ == '__main__':
    app.run_server(debug=True)
