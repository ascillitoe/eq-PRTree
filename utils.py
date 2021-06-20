import numpy as np
import equadratures as eq
import urllib.parse
import re

def strip_tree(polytree):
    '''
    Recurse tree and strip all unnecessary attributes to reduce its size.
    '''
    def _recurse(node):                   
        if node is None: 
            return

        # Strip attributes
        node.pop("data", None)
        del node['poly'].inputs, node['poly'].outputs

        # Go to children
        _recurse(node["children"]["left"])
        _recurse(node["children"]["right"])
    
    # Kick us off...
    _recurse(polytree.tree)
    return polytree

## FOR DEBUGGING
#def print_tree(polytree):
#    '''
#    Recurse tree and print attributes.
#    '''
#    def _recurse(node):                   
#        if node is None: 
#            return
#
#        # Strip attributes
#        print(node)
#        print(dir(node['poly']))
#
#        # Go to children
#        _recurse(node["children"]["left"])
#        _recurse(node["children"]["right"])
#    
#    # Kick us off...
#    _recurse(polytree.tree)


def convert_latex(text):
    def toimage(x):
        if x[1] and x[-2] == r'$':
            x = x[2:-2]
            img = "\n<img src='https://math.now.sh?from={}&color=black&alternateColor=black' style='display: block; margin: 0.5em auto;'/>\n"
            return img.format(urllib.parse.quote_plus(x))
        else:
            x = x[1:-1]
            return r'![](https://math.now.sh?inline={}&color=black&alternateColor=black)'.format(urllib.parse.quote_plus(x))
    return re.sub(r'\${2}([^$]+)\${2}|\$(.+?)\$', lambda x: toimage(x.group()), text)
