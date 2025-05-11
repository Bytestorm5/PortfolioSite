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
    # Receive user response and record with ground truth, display settings, and screen info
    data = request.get_json() or {}
    # Retrieve test parameters and user ID
    user_id = data.get('user_id')
    colors = data.get('colors')
    result = data.get('result')            # user's choice: 'same' or 'different'
    layout = data.get('layout')            # 'joint', 'disjoint', or 'text'
    background = data.get('background')    # 'black', 'white', or 'transparent'
    # Extract screen properties and response time
    screen = data.get('screen', {})
    response_time_ms = data.get('responseTimeMs')
    # Determine true result via perceptual color difference (CIEDE2000)
    colors_list = colors or []
    delta_e = None
    try:
        if len(colors_list) >= 2:
            # Parse and convert to Lab space
            c0 = Color(colors_list[0]).convert('lab')
            c1 = Color(colors_list[1]).convert('lab')
            # Compute delta E using CIEDE2000
            delta_e = c0.delta_e(c1, method='2000')
            # Threshold below which colors are considered the same
            threshold = float(os.environ.get('COLOR_SIMILARITY_THRESHOLD', 1.0))
            true_same = (delta_e <= threshold)
            true_result = 'same' if true_same else 'different'
        else:
            true_result = 'different'
    except Exception:
        # Fallback to exact match of string values
        true_result = 'same' if len(colors_list) >= 2 and colors_list[0] == colors_list[1] else 'different'
    correct = (result == true_result)
    ip = request.remote_addr
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
        'ip': ip,
        'timestamp': timestamp
    }
    collection = get_collection()
    collection.insert_one(doc)
    # Return feedback
    return {'status': 'ok', 'correct': correct, 'trueResult': true_result}