import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app, server
from apps import supervised
from navbar import navbar
from utils import convert_latex

###################################################################
# Homepage text
###################################################################
# Below is a pretty horrendous hack (see apps/flowfield.py for a much cleaner approach). The markdown sections are split here to get the equations nicely centered (can't use divs etc due to convert_latex function). Hopefully one day dash will sort out their latex support.

home_text = r'''
## Polynomial Regression Trees

**Under construction!**

#### References
\[1]: 

\[2]: 
'''

home_text = dcc.Markdown(convert_latex(home_text), dangerously_allow_html=True, style={'text-align':'justify'})

# disclaimer message
final_details = r'''
This app is currently hosted *on the cloud* via [Heroku](https://www.heroku.com). Resources are limited and the app may be slow when there are multiple users. If it is too slow please come back later! 

Please report any bugs to [ascillitoe@effective-quadratures.org](mailto:ascillitoe@effective-quadratures.org).
'''
final_details = dbc.Alert(dcc.Markdown(final_details),
        dismissable=True,is_open=True,color='info',style={'padding-top':'0.4rem','padding-bottom':'0.0rem'})

msg_404 = r'''
**Oooops** 

Looks like you might have taken a wrong turn!
'''

container_404 = dbc.Container([ 
    dbc.Row(
            [
                dcc.Markdown(msg_404,style={'text-align':'center'})
            ], justify="center", align="center", className="h-100"
    )
],style={"height": "90vh"}
)

###################################################################
# Footer
###################################################################
footer = html.Div(
        [
            html.P('App built by Ashley Scillitoe'),
#            html.A(html.P('ascillitoe.com'),href='https://ascillitoe.com'),
            html.P(html.A('ascillitoe.com',href='https://ascillitoe.com')),
            html.P('Copyright © 2021')
        ]
    ,className='footer',id='footer'
)

###################################################################
# App layout (adopted for all sub-apps/pages)
###################################################################
homepage = dbc.Container([home_text,final_details])

app.layout = html.Div(
    [
        dcc.Location(id='url', refresh=True),
        navbar,
        html.Div(homepage,id="page-content"),
        footer,
    ],
    style={'padding-top': '70px'}
)

###################################################################
# Callback to return page requested in navbar
###################################################################
@app.callback(Output('page-content', 'children'),
    Output('footer','style'),
    Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/':
        return homepage, {'display':'block'}
    if pathname == '/supervised':
        return supervised.layout, {'display':'block'}
    else:
        return container_404, {'display':'none'}

###################################################################
# Run server
###################################################################
if __name__ == '__main__':
    app.run_server(debug=True)

