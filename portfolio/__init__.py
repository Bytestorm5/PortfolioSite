from flask import Flask
from dotenv import load_dotenv

load_dotenv()


def create_app():
    app = Flask(__name__)

    from .routes import main_bp
    from tools.color_picker import bp as color_picker_bp
    from tools.righthand_regex import bp as righthand_regex_bp
    from tools.color_similarity import bp as color_similarity_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(color_picker_bp, url_prefix='/tool/color_picker')
    app.register_blueprint(righthand_regex_bp, url_prefix='/tool/righthand_regex')
    app.register_blueprint(color_similarity_bp, url_prefix='/tool/color_similarity')

    return app
