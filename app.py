from flask import Flask, render_template, send_file, redirect, url_for, request
from werkzeug import exceptions as HTTPError
import platform
from nb import load_nb, get_nbs, get_nb_path
from markupsafe import escape

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/projects')
def projects():
    return render_template('projects.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/blog')
def blog():
    return render_template('blog.html', notebooks=get_nbs())

@app.route('/notebook/<nb>')
def notebook(nb:str):
    nb = load_nb(escape(nb))
    if nb == None:
        return redirect(url_for('blog'))
    return render_template('notebook.html', notebook=nb)

@app.route('/download_notebook/<nb>')
def dl_notebook(nb:str):
    path = get_nb_path(escape(nb))
    if path == None:
        raise HTTPError.NotFound("This notebook does not exist")
    else:
        return send_file(
            path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'{escape(nb)}.ipynb'
        )

@app.route('/details/att_s24')
def att_s24():
    return render_template('/work_details/att_s24.html')

@app.route('/tools')
def tools():
    return render_template('tools.html')

@app.route('/tools/righthand_regex')
def righthand_regex():
    return render_template('/tools/righthand_regex.html')

@app.route('/tools/color_picker')
def color_picker():
    return render_template('/tools/color_picker.html')

from coloraide import Color
import itertools
import numpy as np
from scipy.stats import norm

@app.route('/tools_api/color_picker', methods = ["POST"])
@profile
def color_picker_api():
    colors_raw = request.json['colors']
    colors = [Color(c).convert('oklch') for c in colors_raw]
    
    L_mean = np.mean([c._coords[0] for c in colors])
    C_mean = np.mean([c._coords[1] for c in colors])
    L_std = np.std([c._coords[0] for c in colors])
    C_std = np.std([c._coords[1] for c in colors])
    
    for i in range(len(colors)):
        for filter in ['protan', 'deutan', 'tritan']:
            colors.append(colors[i].filter(filter).fit('srgb'))
    
    def objective(LCH):
        test_color = Color('oklch', LCH).fit('srgb')
        diff = min([test_color.delta_e(c, method='ok') for c in colors])
        
        L_prob = norm.pdf(LCH[0], L_mean, L_std)
        C_prob = norm.pdf(LCH[1], C_mean, C_std)

        # Combine the probabilities for weighting
        weight = L_prob * C_prob

        return diff - 5*(1 - weight)
    
    search_space = itertools.product(
        np.linspace(0, 1, 5), # L
        np.linspace(0, 0.37, 5), # C
        np.linspace(0, 360, 36), # H        
    )
    
    best_color = max(search_space, key=objective)
    return {
        "color": best_color,
        "fitness": objective(best_color)
    }

if __name__ == '__main__':
    if platform.system() == 'Windows':
        app.run(debug=True, port=5000)
    else:
        app.run(port=80, host='0.0.0.0')