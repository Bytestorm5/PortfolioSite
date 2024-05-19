from flask import Flask, render_template, send_file
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
    return render_template('notebook.html', notebook=load_nb(escape(nb)))

@app.route('/download_notebook/<nb>')
def dl_notebook(nb:str):
    path = get_nb_path(escape(nb))
    if not path:
        raise HTTPError.NotFound("This notebook does not exist")
    else:
        return send_file(
            path,
            mimetype='application/octet-stream',
            as_attachment=True,
            download_name=f'{escape(nb)}.ipynb'
        )

if __name__ == '__main__':
    if platform.system() == 'Windows':
        app.run(debug=True, port=5000)
    else:
        app.run(port=80, host='0.0.0.0')