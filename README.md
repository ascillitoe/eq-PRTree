# eq-PRTree

This is a python [Dash](https://plotly.com/dash/) app, (soon to be) accesible at [PRTree.ascillitoe.com](https://PRTree.ascillitoe.com/). It demonstrates the utility of polynomial regression trees (PRTrees); a new type of model tree for supervised machine learning and polynomial chaos applications. By building decision trees with orthogonal polynomials at the leaf nodes, PRTrees can achieve the same accuracy with a significantly shallower tree compared to a standard decision tree. In fact, the predicitve accuracy of a PRTree is often competitive with ensemble methods such as random forests and gradient boosted tree. Moreover, the shallow tree depth, combined with the polynomials' sensitivity indices, yields a readily interpretable model. 

The PRTrees are obtained with the [PolyTree](https://equadratures.org/_documentation/polytree.html) module in the [equadratures](https://equadratures.org/) package. The app is hosted on the cloud via a [Heroku](https://www.heroku.com/about) dyno. Memory-caching is performed with [Flask-Caching](https://flask-caching.readthedocs.io/en/latest/), [pylibmc](https://pypi.org/project/pylibmc/), and [MemCachier](https://www.memcachier.com/).

### Installation
The easiest way to run the app is to simply go to [PRTree.ascillitoe.com](https://PRtree.ascillitoe.com/)! 

Alternatively, if you want to run locally, you must first install the full dash stack, and a number of other packages. The full list of requirements for local running can be installed with:

```console
python -m pip install dash dash_core_components dash_html_components dash_daq 
python -m pip install flask_caching pylibmc plotly numpy pandas equadratures>=9.1.0 jsonpickle func-timeout requests>=2.11.1
```

I recommend performing the above in a virtual enviroment such as virtualenv. After installing requirements, the app can be launched with:

```console
python index.py
```
