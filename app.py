from flask import Flask, render_template, send_file, redirect, url_for, request
import platform
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    if platform.system() == 'Windows':
        app.config["TEMPLATES_AUTO_RELOAD"] = True
        app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
        app.run(debug=True, port=5000)
    else:
        app.run(port=80, host='0.0.0.0')