"""Differentiated Color Picker blueprint."""
from flask import Blueprint, render_template, request
from coloraide import Color
import itertools
import numpy as np
from scipy.stats import norm

bp = Blueprint('color_picker', __name__)

@bp.route('/', methods=['GET'])
def index():
    return render_template('tools/color_picker.html')

@bp.route('/api', methods=['POST'])
def api():
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
            if k is None:
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