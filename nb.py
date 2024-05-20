from nbconvert import HTMLExporter
import nbformat
import os
import time
import pandas as pd

nb_directory = pd.read_csv(os.path.join('notebooks', 'directory.csv')).to_numpy()

def convert_notebook_to_html(notebook_path):
    with open(notebook_path, encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    html_exporter = HTMLExporter()
    (body, resources) = html_exporter.from_notebook_node(notebook)
    return body

def load_nb(name: str):
    if not name in nb_directory[:, 1]:
        return None
    
    nb_path = os.path.join('notebooks', 'nbs', name + '.ipynb')    
    
    if not os.path.exists(nb_path):
        raise FileNotFoundError(f"Notebook {name} not found.")
    
    cache_path = os.path.join('notebooks', 'cache', name + '.html')
    
    nb_last_modified = os.path.getmtime(nb_path)
    
    if os.path.exists(cache_path) and os.path.getmtime(cache_path) >= nb_last_modified:
        with open(cache_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return content
    else:
        html_content = convert_notebook_to_html(nb_path)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return html_content

def get_nb_path(name):
    if not name in nb_directory[:, 1]:
        return None
    nb_path = os.path.join('notebooks', 'nbs', name + '.ipynb')
    if os.path.exists(nb_path):
        return nb_path
    else:
        return None

def get_nbs():
    return nb_directory