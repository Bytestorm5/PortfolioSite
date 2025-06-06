from flask import Blueprint, render_template, redirect, url_for, send_file
from werkzeug import exceptions as HTTPError
from markupsafe import escape
from nb import load_nb, get_nbs, get_nb_path

main_bp = Blueprint('main', __name__, template_folder='../templates')

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/projects')
def projects():
    return render_template('projects.html')

@main_bp.route('/contact')
def contact():
    return render_template('contact.html')

@main_bp.route('/blog')
def blog():
    return render_template('blog.html', notebooks=get_nbs())

@main_bp.route('/notebook/<nb>')
def notebook_view(nb: str):
    nb_html = load_nb(escape(nb))
    if nb_html is None:
        return redirect(url_for('main.blog'))
    return render_template('notebook.html', notebook=nb_html)

@main_bp.route('/download_notebook/<nb>')
def download_notebook(nb: str):
    path = get_nb_path(escape(nb))
    if path is None:
        raise HTTPError.NotFound("This notebook does not exist")
    return send_file(path, mimetype='application/octet-stream', as_attachment=True, download_name=f'{escape(nb)}.ipynb')

@main_bp.route('/details/att_s24')
def att_s24():
    return render_template('work_details/att_s24.html')

@main_bp.route('/tools')
def tools_page():
    return render_template('tools.html')
