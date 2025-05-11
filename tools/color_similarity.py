"""Color Similarity Test blueprint."""
import os
from flask import Blueprint, render_template, request
from pymongo import MongoClient
from datetime import datetime

bp = Blueprint('color_similarity', __name__)

def get_collection():
    uri = os.environ.get('MONGO_URI')
    if not uri:
        raise RuntimeError('MONGO_URI environment variable not set')
    client = MongoClient(uri)
    # Use database specified in the URI
    db = client["PortfolioSite"]
    # Collection for storing similarity results
    return db["color_similarity"]

@bp.route('/', methods=['GET'])
def index():
    return render_template('tools/color_similarity.html')

@bp.route('/api', methods=['POST'])
def api():
    # Receive user response and record with ground truth
    data = request.get_json() or {}
    colors = data.get('colors')
    result = data.get('result')  # user's choice: 'same' or 'different'
    # Determine true result: identical strings imply same color
    true_result = 'same' if (colors and len(colors) >= 2 and colors[0] == colors[1]) else 'different'
    correct = (result == true_result)
    ip = request.remote_addr
    timestamp = datetime.utcnow()
    doc = {
        'colors': colors,
        'result': result,
        'true_result': true_result,
        'correct': correct,
        'ip': ip,
        'timestamp': timestamp
    }
    collection = get_collection()
    collection.insert_one(doc)
    # Return feedback
    return {'status': 'ok', 'correct': correct, 'trueResult': true_result}