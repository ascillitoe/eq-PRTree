import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import dash_daq as daq
import dash_table
from dash_table.Format import Format, Scheme, Trim
from dash.dependencies import Input, Output, State, ALL
from flask_caching import Cache
import plotly.graph_objs as go

import os
import base64
import io
import re
import pickle
import jsonpickle
import math
import numpy as np
import pandas as pd
import equadratures as eq
from sklearn import tree
from sklearn.datasets import fetch_california_housing
from utils import convert_latex
from func_timeout import func_timeout, FunctionTimedOut
import dash_interactive_graphviz as dig
import pydot

from app import app

###################################################################
# Setup cache (simple cache if locally run, otherwise configured
# to use memcachier on heroku)
###################################################################
cache_servers = os.environ.get('MEMCACHIER_SERVERS')
if cache_servers == None:
    # Fall back to simple in memory cache (development)
    cache = Cache(app.server,config={'CACHE_TYPE': 'SimpleCache'})
else:
    cache_user = os.environ.get('MEMCACHIER_USERNAME') or ''
    cache_pass = os.environ.get('MEMCACHIER_PASSWORD') or ''
    cache = Cache(app.server,
        config={'CACHE_TYPE': 'SASLMemcachedCache',
                'CACHE_MEMCACHED_SERVERS': cache_servers.split(','),
                'CACHE_MEMCACHED_USERNAME': cache_user,
                'CACHE_MEMCACHED_PASSWORD': cache_pass,
                'CACHE_OPTIONS': { 'behaviors': {
                    # Faster IO
                    'tcp_nodelay': True,
                    # Keep connection alive
                    'tcp_keepalive': True,
                    # Timeout for set/get requests
                    'connect_timeout': 2000, # ms
                    'send_timeout': 750 * 1000, # us
                    'receive_timeout': 750 * 1000, # us
                    '_poll_timeout': 2000, # ms
                    # Better failover
                    'ketama': True,
                    'remove_failed': 1,
                    'retry_timeout': 2,
                    'dead_timeout': 30}}})

###################################################################
# Collapsable more info card
###################################################################
info_text = r'''
Upload your data in *.csv* format using the **Load Data** card. Take note of the following:
- The data must in standard *wide-format* i.e. with each row representing an observation/sample. 
- There must be no NaN's or empty cells.
- For computational cost purposes datasets are currently capped at $N=2000$ rows and $d=20$ input dimensions. For guidance on handling larger datasets checkout the [docs](https://equadratures.org/) or [get in touch](https://discourse.equadratures.org/).
- Particularly when higher polynomial orders and/or high maximum tree depths are selected, monitor test accuracy to check for *over-fitting*.
- If the accuracy is poor, try to vary the polynomial order and maximum tree depth.
- Due to computational cost considerations, ordinary least squares regression is used for polynomial fitting here. This may lead to poor polynomial fits if order > 1 polynomials are used with noisy or discontinous data. When using [equadratures](https://equadratures.org/), adding some regularisation with `poly_method='elastic-net'` may help in such cases (see the [PolyTree](https://equadratures.org/_documentation/polytree.html) and [elastic net](https://equadratures.org/_documentation/solver.html#equadratures.solver.elastic_net) docs).
'''

info = html.Div(
    [
    dbc.Button("More Information",color="primary",id="data-info-open",className="py-0"),
    dbc.Modal(
        [
            dbc.ModalHeader(dcc.Markdown('**More Information**')),
            dbc.ModalBody(dcc.Markdown(convert_latex(info_text),dangerously_allow_html=True)),
            dbc.ModalFooter(dbc.Button("Close", id="data-info-close", className="py-0", color='primary')),
        ],
        id="data-info",
        scrollable=True,size='lg'
    ),
    ]
)


###################################################################
# Load data card
###################################################################
data_select = dbc.Form(
    [
    dbc.Label('Select dataset',html_for='data-select'),
    dcc.Dropdown(id="data-select",
    options=
        [
        {"label": "Upload my own", "value":"upload"},
        {"label": "Airfoil noise", "value":"airfoil"},
        {"label": "California housing prices", "value":"cali"},
        {"label": "Temperature probes", "value":"probes"},
        ],
        value="upload",placeholder="Upload my own",clearable=False,searchable=False)
    ]
)

upload = dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select',style={'font-weight':'bold','color':'var(--primary)'}),
            ' CSV/Excel file'
        ]),
        # Don't allow multiple files to be uploaded
        multiple=False, disabled=False,
        style = {
                'width': '100%',
                'height': '50px',
                'lineHeight': '50px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '5px',
                }

    )

qoi_input = dbc.Row(
    [
    dbc.Col(
        dbc.FormGroup(
            [
                dbc.Label('Select output variable', html_for="qoi-select"),
                dcc.Dropdown(id="qoi-select",searchable=False)
            ],
        ),width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Input dimensions:', html_for="report-Xd",width=9),
                dbc.Col(html.Div(id='report-Xd'),width=3)
                ], row=True
            ),
            dbc.FormGroup(
                [
                dbc.Label('Output dimensions:', html_for="report-yd",width=9),
                dbc.Col(html.Div(id='report-yd'),width=3)
                ], row=True
            ),
            ]
        ), width=4
    ),
    dbc.Col(
        dbc.Form(
            [
            dbc.FormGroup(
                [
                dbc.Label('Training samples:', html_for="report-train",width=9),
                dbc.Col(html.Div(id='report-train'),width=3)
                ], row=True
            ),
            dbc.FormGroup(
                [
                dbc.Label('Test samples:', html_for="report-test",width=9),
                dbc.Col(html.Div(id='report-test'),width=3)
                ], row=True
            ),
            ]
        ), width=4
    ),

    ]
)

data_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Load Data**')),
        dbc.CardBody(
            [
            dbc.Row(
                [
                dbc.Col(data_select,width=4),
                dbc.Col(upload,width=5),
                dbc.Col(
                    dbc.Alert('Error loading file!',id='upload-error',color='danger',is_open=False,style={'height':'40px','margin':'10px'}),
                width=3),
                ],align='start'
            ),
            dbc.Row(dbc.Col(
                dbc.Alert('Click bin icons to delete columns as necessary',id='upload-help',color='info',
                    is_open=False,dismissable=True,style={'margin-top':'0.4rem'}),
                width=5)
            ),
            dbc.Row(dbc.Col(
                dash_table.DataTable(data=[],columns=[],id='upload-data-table',
                style_table={'overflowX': 'auto','overflowY':'auto','height':'35vh'}, #change height to maxHeight to get table to only take up space when populated.
                editable=True,fill_width=False,page_size=20)
                ,width=12),style={'margin-top':'10px'}),
            dbc.Row(
                [
                dbc.Col(qoi_input,width=12),
                ]
            )
                
            # TODO - either summary of example dataset, or table of data metrics appear here
            ]
        )
    ]
)

###################################################################
# Settings card
###################################################################
order_slider = dbc.FormGroup(
    [
        dbc.Label('Polynomial order', html_for="order-slider"),
        dcc.Slider(id='order-slider',min=1, max=3,value=1,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ],
)

maxdepth_slider = dbc.FormGroup(
    [
        dbc.Label('Maximum tree depth', html_for="maxdepth-slider"),
        dcc.Slider(id='maxdepth-slider',min=1, max=4,value=1,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ], 
)

traintest_slider = dbc.FormGroup(
    [
        dbc.Label('Test split (%)', html_for="traintest-slider"),
        dcc.Slider(id='traintest-slider',min=0, max=50,value=0,
            tooltip = { 'always_visible': True, 'placement': 'bottom' }
        )
    ],
)

settings_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Training**')),
        dbc.CardBody(
            [
            dbc.Row(dbc.Col(order_slider)),
            dbc.Row(dbc.Col(maxdepth_slider)),
            dbc.Row(dbc.Col(traintest_slider)),
            dbc.Row(
                [
                dbc.Col(dbc.Button("Compute", id="compute", color="primary"),width='auto'),
                dbc.Col(dbc.Spinner(html.Div(id='compute-finished'),color="primary"),width=2),
                dbc.Col(dbc.Alert(id='compute-warning',color='danger',is_open=False),width='auto'),
                ], justify='start',align='center'
            ),
            dbc.Row(dbc.Col(dcc.Markdown(id='r2-train',dangerously_allow_html=True)),style={'margin-top':'10px'}),
            dbc.Row(dbc.Col(dcc.Markdown(id='r2-test',dangerously_allow_html=True))),
            ]
        )
    ]
)

###################################################################
# Accuracy card
###################################################################
accuracy_dropdown = dbc.FormGroup(
    [
        dbc.Label('Accuracy Metric', html_for="accuracy-select",width=5),
        dbc.Col(dcc.Dropdown(id="accuracy-select",options=[
            {'label': 'R2 score', 'value': 'r2'},
            {'label': 'Adjusted R2 score', 'value': 'adjusted_r2'},
            {'label': 'MAE', 'value': 'mae'},
            {'label': 'RMSE', 'value': 'rmse'},
            ],
        value='r2',clearable=False,searchable=False),width=7
        )
    ],
    id='method-select', row=True
)

table_header = [
    html.Thead(html.Tr([html.Th("Model"), html.Th("Train"), html.Th("Test")]))
]

row1 = html.Tr([html.Td("Decision Tree"),html.Td(id='DT-train'), html.Td(id='DT-test')])
row2 = html.Tr([html.Td("PRTree"),html.Td(id='PT-train'), html.Td(id='PT-test')])

table_body = [html.Tbody([row1, row2])]

accuracy_table = dbc.Table(table_header + table_body, bordered=True, hover=True)

accuracy_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Accuracy**')),
        dbc.CardBody(
            [
            accuracy_dropdown,
            accuracy_table,
            ]
        )
        ], style={'margin-top':'10px'}
)

###################################################################
# Tree viz card
###################################################################
graphviz = html.Div(
        dig.DashInteractiveGraphviz(id="tree-graph"),
        style={'position':'relative','width':'100%','height':'72vh'}
)

select_tree = dcc.Dropdown(id="tree-select",options=[
            {'label': 'Polynomial Tree', 'value': 'PT'},
            {'label': 'Decision Tree', 'value': 'DT'},
            ],
        value='PT',clearable=False,searchable=False
)

sobol_plot = dcc.Graph(
        figure={},id="sobol-plot",
        style={'height':'70vh','width':'inherit'}
)

sobol_msg = r'''
**Click on a node** to view its polynomial's sensitivity indices (Sobol' indices)!
'''

viz_card = dbc.Card(
    [
        dbc.CardHeader(dcc.Markdown('**Visualise Tree**')),
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dbc.Row(dbc.Col(select_tree,width=6)),
                                dbc.Row(dbc.Col(graphviz,width=12))
                            ],width=8
                        ),
                        dbc.Col(
                            [
                                dcc.Markdown(sobol_msg),
                                sobol_plot

                            ],width=4, id='sobol-col'
                        )
                    ]
                )
            ]
        )
    ]#, style={'height':'72vh'}
)

###################################################################
# Timout warning
###################################################################
timeout_msg = dcc.Markdown(r'''
**Timeout!**

Sorry! The computation timed out due to the 30 second time limit imposed by the heroku server. 

You can try:
- Lowering the polynomial order and/or maximum depth of the PRTree.
- Reducing the number of rows in your dataset and/or dimensions in your dataset.
- Coming back later, when the server might be less busy.
''')

timeout_warning = dbc.Modal(
        dbc.ModalBody(timeout_msg, style={'background-color':'rgba(160, 10, 0,0.2)'}),
    id="timeout",
    is_open=False,
)

###################################################################
# The overall app layout
###################################################################
layout = dbc.Container(
    [
    html.H2("Supervised Machine Learning"),
    dbc.Row(
        [
            dbc.Col(dcc.Markdown('This app constructs polynomial regression trees (PRTrees) in a supervised learning manner. Upload your own data, or choose an example dataset.'),width='auto'),
            dbc.Col(info,width='auto')
            ], align='center', style={'margin-bottom':'10px'}
    ),
    dbc.Row(
        [
        dbc.Col(data_card,width=12),
        ]
    ),
    dbc.Row(
        [
            dbc.Col(viz_card,width=9),
            dbc.Col(
                [
                    dbc.Row(dbc.Col(settings_card,width=12)),
                    dbc.Row(dbc.Col(accuracy_card,width=12)),
                ], width=3
            ),
        ],style={'margin-top':'10px'},
    ),
    dcc.Store(id='dt-data'),
    dcc.Store(id='pt-data'),
#    tooltips,
    timeout_warning
    ],
    fluid = True
)


###################################################################
# Callbacks
###################################################################
# More info collapsable
@app.callback(
    Output("data-info", "is_open"),
    [Input("data-info-open", "n_clicks"), Input("data-info-close", "n_clicks")],
    [State("data-info", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# Show upload interface callback
@app.callback(
    Output("upload-data", "disabled"),
    [Input("data-select", "value")],
)
def toggle_upload_box(option):
    if option == 'upload':
        return False
    else:
        return True


# Load csv file callback
@app.callback(Output('upload-data-table', 'data'),
        Output('upload-data-table', 'columns'),
        Output('upload-error', 'is_open'),
        Output('upload-help', 'is_open'),
        Input('upload-data', 'contents'),
        Input("data-select", "value"),
        State('upload-data', 'filename'),
        prevent_initial_call=True)
def load_csv(content, data_option, filename):
    df = None
    if data_option == 'upload':
        # Only load csv once button pressed
        changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
        if 'upload-data' in changed_id:
            # Parse csv
            content_type, content_string = content.split(',')
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
                return [],[],True,False
            bin_msg = True
    # temperature probe dataset
    elif data_option == 'probes':
        data = eq.datasets.load_eq_dataset('probes')
        data = np.hstack([data['X'],data['y2']])
        cols = ['Hole ellipse','Hole fwd/back','Hole angle','Kiel lip','Kiel outer','Kiel inner','Hole diam.','Recovery ratio objective']
        df = pd.DataFrame(data=data, columns=cols)
        bin_msg = False
    # Airfoil
    elif data_option == 'airfoil':
        df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat', 
                sep="\t", names = ['Freq','AOA','ChordLength','FSV','Suction','SPL'])
        bin_msg = False
    # Cali housing
    elif data_option == 'cali':
        cali = fetch_california_housing()
        df = pd.DataFrame(data=cali['data'], columns=cali['feature_names'])
        df = df.sample(n = 2000, replace = False, random_state=42) 
        bin_msg = False

    # Create a datatable
    if df is not None:
        data=df.to_dict('records')
        columns=[{'name': i, 'id': i, 'deletable': True,'type':'numeric','format':Format(precision=4,trim=Trim.yes)} for i in df.columns]
        return data,columns,False,bin_msg
    else:
        return [], [], False, False

# Populate qoi options
@app.callback(Output('qoi-select','options'),
        Output('qoi-select','value'),
        Input('upload-data-table', 'columns'),
        State("data-select", "value"),
        prevent_initial_call=True)
def populate_qoi(columns,data_option):
    if data_option == 'upload':
        options = [{'label': i['name'], 'value': i['name']} for i in columns]
        value = None
    else:
        output = columns[-1]['name']
        options = [{'label': output, 'value': output}]
        value = output
    return options, value

##################################################################
# Function to compute trees
###################################################################
# Compute polytree
@cache.memoize(timeout=600)
def compute_trees_memoize(X_train, y_train, max_depth, order):
    # Decision tree fitting
    dt = tree.DecisionTreeRegressor(max_depth=max_depth,criterion='mse',min_samples_leaf=2)
    dt = dt.fit(X_train,y_train)

    # Polytree fitting
    pt = eq.polytree.PolyTree(splitting_criterion='loss_gradient',order=order,max_depth=max_depth)
    pt.fit(X_train,y_train)

    return jsonpickle.encode(dt), jsonpickle.encode(pt)

# callback to compute trees
@app.callback(Output('compute-finished','children'),
        Output('compute-warning','is_open'),
        Output('compute-warning','children'),
        Output('dt-data','data'),
        Output('pt-data','data'),
        Output('DT-train','children'),
        Output('DT-test','children'),
        Output('PT-train','children'),
        Output('PT-test','children'),
        Output("timeout", "is_open"),
        Input('compute', 'n_clicks'),
        Input('upload-data-table', 'data'),
        Input('upload-data-table', 'columns'),
        Input('qoi-select','value'),
        Input('order-slider','value'),
        Input('maxdepth-slider','value'),
        Input('traintest-slider','value'),
        Input('accuracy-select','value'),
        prevent_initial_call=True)
def compute_trees(n_clicks,data,cols,qoi,order,max_depth,test_split, metric):
    # Compute subspace (if button has just been pressed)
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'compute' in changed_id:
        # Check qoi selected
        if qoi is None:
            return None, True, 'Output variable not selected!', None, None, None, None, None, None, False
        else:
            # Parse data to dataframe
            df = pd.DataFrame.from_records(data)
            # Check for missing values and NaN
            problem = df.isnull().values.any() 
            if problem:
                return None, True, 'Missing/NaN values in data', None, None, None, None, None, None, False

            # Get X and y
            y = df.pop(qoi).to_numpy()
            X = df.to_numpy()
            
            # Train/test split
            test_split /= 100
            X_train, X_test, y_train, y_test = eq.datasets.train_test_split(X, y,
                                   train=float(1-test_split),random_seed=42)
 
            # Compute trees
            try:
                dt_pickled, pt_pickled = func_timeout(28,compute_trees_memoize,args=(X_train, y_train, max_depth, order))
            except FunctionTimedOut:
                return None, False, None, None, None,None, None, None, None, True

            # Compute scores
            dt = jsonpickle.decode(dt_pickled)
            pt = jsonpickle.decode(pt_pickled)

            # Training
            dt_train = '%.3g' %eq.datasets.score(y_train, dt.predict(X_train), metric=metric, X=X_train )
            pt_train = '%.3g' %eq.datasets.score(y_train, pt.predict(X_train), metric=metric, X=X_train )
            # Test
            if y_test.size > 0:
                dt_test = '%.3g' %eq.datasets.score(y_test, dt.predict(X_test), metric=metric, X=X_test )
                pt_test = '%.3g' %eq.datasets.score(y_test, pt.predict(X_test), metric=metric, X=X_test )
            else:
                dt_test = None
                pt_test = None

            # Return data
            return None, False, None, dt_pickled, pt_pickled, dt_train, dt_test, pt_train, pt_test, False

    return None, False, None, None, None, None, None, None, None, False

###################################################################
# graphviz graphs callbacks
###################################################################
# Generate treee graph
@app.callback(Output('tree-graph','dot_source'),
    Input('dt-data','data'),
    Input('pt-data','data'),
    Input('upload-data-table', 'columns'),
    Input('qoi-select','value'),
    Input('tree-select','value'),
    Input("tree-graph", "selected_node"))
def create_tree_graph(dt_pickled,pt_pickled,cols,qoi,tree_select,selected_node):
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'tree-select' in changed_id:
        selected_node = None # Reset if just changed from DT to PT otherwise errors 

    if dt_pickled is None or pt_pickled is None:
        return None
    else:
        features = [col['name'] for col in cols]
        features.remove(qoi)

        if tree_select == 'DT':
            dt = jsonpickle.decode(dt_pickled)        
            dot_source = tree.export_graphviz(dt, out_file=None, 
                    feature_names=features,
                    filled=False, rounded=True,  
                    special_characters=True)  
            graph = pydot.graph_from_dot_data(dot_source)[0]

        elif tree_select == 'PT':
            pt = jsonpickle.decode(pt_pickled)
            dot_source = pt.get_graphviz(feature_names=features,file_name='source')
            graph = pydot.graph_from_dot_data(dot_source)[0]

        # Reset stylings so pt and dt match
        for node in graph.get_nodes():
            clean_node_label(node,tree_select)
            node.set('style','filled, rounded')
            node.set('fillcolor','white')
        for edge in graph.get_edges():
            edge.set('fillcolor','black')

        # Highlight selected node (for pt only)
        if selected_node is not None and tree_select=='PT':

            node = graph.get_node(str(selected_node))
            print(node)
            if len(node)==0: # This occurs when a selected node no longer exists (i.e. because max_depth reduced after selecting node)
                pass
            else:
                node[0].set('fillcolor','#87CEFA')
        print(graph)
        return graph.to_string()

def clean_node_label(node,tree):
    string = node.get('label')
    if string is None:
        pass
    else:
        if tree == 'PT':
            items = string.split(r'\n')
            items = [item.strip() for item in items]
            items = [item.replace('"','') for item in items]
            items = [item.replace('n_samples','samples') for item in items]
            items = [item.replace('loss','mse') for item in items]
            mse = float(re.findall(r"[-+]?\d*\.\d+|\d+", items[-1])[0])
            items[-1] = 'mse = %.3f' %mse
            del items[0]
        elif tree == 'DT':
            items = string.split(r'<br/>')
            items = [item.strip() for item in items]
            items = [item.replace('"','') for item in items]
            items = [item.replace('<','') for item in items]
            items = [item.replace('>','') for item in items]
            tmp = items[1]
            items[1] = items[2]
            items[2] = tmp
        string = r'\n'.join(items)
        node.set('label',string)

###################################################################
# Plot callback
###################################################################
@app.callback(Output('sobol-plot', 'figure'),
    Output('sobol-col','style'),
    Input('pt-data','data'),
    Input('upload-data-table', 'columns'),
    Input('qoi-select','value'),
    Input('tree-select','value'),
    Input("tree-graph", "selected_node"))

def display_sobol_plot(pt_pickled,cols,qoi,tree_select,selected_node):
    # layout
    layout={"xaxis": {"title": r'$S_i$'},'margin':{'t':0,'r':0,'l':0,'b':60},
            'paper_bgcolor':'white','plot_bgcolor':'white','autosize':True}
    fig = go.Figure(layout=layout)
    fig.update_xaxes(color='black',linecolor='black',showline=True,tickcolor='black',ticks='outside',range=[0,1])
    fig.update_yaxes(color='black',linecolor='black',showline=True,showticklabels=True)
    
    # Parse results
    if tree_select == 'PT':
        style = {'display':'block'}
        if pt_pickled is not None and selected_node is not None:
            # Get 1st order Sobol indices for selected node
            pt = jsonpickle.decode(pt_pickled)
            node = pt.get_node(int(selected_node))
            if node is None: # This occurs when a selected node no longer exists (i.e. because max_depth reduced after selecting node)
                return fig, style
            poly = node['poly']
#            sobol = list(poly.get_sobol_indices(1).values())
            sobol = poly.get_total_sobol_indices()
            names = [col['name'] for col in cols]
            names.remove(qoi)
            names = [name + '   ' for name in names]

            # Plot as bar chart
            fig.add_trace(go.Bar(x=sobol, y=names, marker_color='LightSkyBlue',
                    marker_line_width=2,marker_line_color='black',orientation='h'))
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide',yaxis_tickangle=-45)
        
    else:
        style = {'display':'none'}

    return fig, style

###################################################################
# Limit data size
###################################################################
# Callback to limit rows
@app.callback(Output('compute','disabled'),  
    Output('report-train','children'),
    Output('report-test','children'),
    Output('report-train','style'),
    Output('report-test','style'),
    Output('report-Xd','children'),
    Output('report-yd','children'),
    Output('report-Xd','style'),
    Output('report-yd','style'),
    Input('traintest-slider','value'),
    Input('upload-data-table', 'data'),
    Input('upload-data-table', 'columns'),
    Input('qoi-select','value'),
    prevent_initial_call=True)
def check_size(test_split,data,cols,qoi):
    MAX_ROWS = 2000
    MAX_D = 20

    # Number of rows
    N = len(data)
    test_split = float(test_split/100)
    Ntest = math.ceil(N*test_split)
    Ntrain = N-Ntest

    # Number of dims
    d = len(cols)
    if qoi is None:
        Xd = d
        yd = 0
    else:
        Xd = d-1
        yd = 1

    # check dataset size
    toobig = False
    Ncolor = 'black'
    Dcolor = 'black'
    if Ntrain > MAX_ROWS: 
        toobig = True
        Ncolor = 'red'
    if d > MAX_D: 
        toobig = True
        Dcolor = 'red'

    return toobig, str(Ntrain), str(Ntest), {'color':Ncolor}, {'color':'black'}, str(Xd), str(yd), {'color':Dcolor}, {'color':'black'}
