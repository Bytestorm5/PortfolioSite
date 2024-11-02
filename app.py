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
def color_picker_api():
    colors_raw = request.json['colors']
    colors = [Color(c).convert('oklch') for c in colors_raw]
    fcolors: dict[str, list] = {None: colors}
    
    L_mean = np.mean([c._coords[0] for c in colors])
    C_mean = np.mean([c._coords[1] for c in colors])
    L_std = np.std([c._coords[0] for c in colors]) or 1e-5
    C_std = np.std([c._coords[1] for c in colors]) or 1e-5
    max_L_prob = norm.pdf(L_mean, L_mean, L_std)
    max_C_prob = norm.pdf(C_mean, C_mean, C_std)
    
    for filter in ['protan', 'deutan', 'tritan']:
        fcolors[filter] = []
        for i in range(len(colors)):
            fcolors[filter].append(colors[i].filter(filter).fit('srgb'))
    
    def objective(LCH, disable_prob=False):
        test_color = Color('oklch', LCH).fit('srgb')
        diff = 9999999
        for k in fcolors.keys():
            if k == None:
                diff = min([diff] + [test_color.delta_e(c, method='ok') for c in fcolors[k]])
            else:
                ftest = test_color.filter(k)
                diff = min([diff] + [ftest.delta_e(c, method='ok') for c in fcolors[k]])
        if disable_prob:
            return diff
        
        L_prob = abs(norm.pdf(LCH[0], L_mean, L_std)) / max_L_prob
        C_prob = abs(norm.pdf(LCH[1], C_mean, C_std)) / max_C_prob

        weight = L_prob * C_prob
        weight = np.exp(-np.power(5*(weight - 0.5), 2))

        return diff*weight
    
    search_space = itertools.product(
        np.linspace(0, 1, 5), # L
        np.linspace(0, 0.37, 5), # C
        np.linspace(0, 360, 36), # H        
    )
    
    best_color = max(search_space, key=objective)
    best_color_obj = Color('oklch', best_color)
    pure_fit = objective(best_color, disable_prob=True)
    prob = objective(best_color) / pure_fit
    return {
        "color_raw": best_color,
        "color": best_color_obj.convert('srgb').to_string(hex=False),
        "color_hex": best_color_obj.convert('srgb').to_string(hex=True),
        "fitness": f"{objective(best_color):.2f}",
        "pure_fitness": f"{pure_fit:.2f}",
        "prob_fitness": f"{prob:.4f}"
    }

if __name__ == '__main__':
    if platform.system() == 'Windows':
        app.run(debug=True, port=5000)
    else:
        app.run(port=80, host='0.0.0.0')