"""Color Similarity Test blueprint."""
import os
from flask import Blueprint, render_template, request
from pymongo import MongoClient
from datetime import datetime
from coloraide import Color

bp = Blueprint('color_similarity', __name__)

def get_collection():
    uri = os.environ.get('MONGO_URI')
    if not uri:
        raise RuntimeError('MONGO_URI environment variable not set')
    client = MongoClient(uri)
    # Use database specified in the URI (from connection string)
    db = client["PortfolioSite"]
    # Collection for storing similarity results
    return db["color_similarity"]

@bp.route('/', methods=['GET'])
def index():
    return render_template('tools/color_similarity.html')

@bp.route('/api', methods=['POST'])
def api():
    data = request.get_json() or {}
    user_id = data.get('user_id')
    colors = data.get('colors')
    result = data.get('result')
    layout = data.get('layout')
    background = data.get('background')
    # Extract screen properties and response time
    screen = data.get('screen', {})
    response_time_ms = data.get('responseTimeMs')
    colors_list = colors or []
    delta_e = None
    try:
        if len(colors_list) >= 2:
            c0 = Color(colors_list[0]).convert('lab')
            c1 = Color(colors_list[1]).convert('lab')
            delta_e = c0.delta_e(c1, method='2000')
            threshold = float(os.environ.get('COLOR_SIMILARITY_THRESHOLD', 1.5))
            true_same = (delta_e <= threshold)
            true_result = 'same' if true_same else 'different'
        else:
            true_result = 'different'
    except Exception:
        # Fallback to exact match of string values
        true_result = 'same' if len(colors_list) >= 2 and colors_list[0] == colors_list[1] else 'different'
    correct = (result == true_result)
    # Compute analysis under different colorblind filters
    analysis = {}
    try:
        if len(colors_list) >= 2:
            for filt in (None, 'protan', 'deutan', 'tritan'):
                # Parse base colors
                c0 = Color(colors_list[0])
                c1 = Color(colors_list[1])
                if filt:
                    c0 = c0.filter(filt)
                    c1 = c1.filter(filt)
                # Convert to displayable sRGB and fit gama
                c0s = c0.convert('srgb').fit('srgb')
                c1s = c1.convert('srgb').fit('srgb')
                hex0 = c0s.to_string(hex=True)
                hex1 = c1s.to_string(hex=True)
                d0 = c0.convert('lab')
                d1 = c1.convert('lab')
                de_val = d0.delta_e(d1, method='2000')
                key = filt or 'normal'
                analysis[key] = { 'colors': [hex0, hex1], 'delta_e': de_val }
    except Exception:
        # If analysis fails, leave empty or minimal
        pass
    timestamp = datetime.utcnow()
    doc = {
        'user_id': user_id,
        'colors': colors,
        'result': result,
        'true_result': true_result,
        'correct': correct,
        'layout': layout,
        'background': background,
        'screen': screen,
        'delta_e': delta_e,
        'response_time_ms': response_time_ms,
        'analysis': analysis,
        'timestamp': timestamp
    }
    collection = get_collection()
    collection.insert_one(doc)
    # Return feedback
    return {
        'status': 'ok',
        'correct': correct,
        'trueResult': true_result,
        'analysis': analysis
    }