"""Righthand Regex Tester blueprint."""
from flask import Blueprint, render_template

bp = Blueprint('righthand_regex', __name__)

@bp.route('/', methods=['GET'])
def index():
    return render_template('tools/righthand_regex.html')