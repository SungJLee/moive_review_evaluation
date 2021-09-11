from flask import Flask

def create_app():
    app = Flask(__name__)

    from .views import evaluation_views, main_views
    app.register_blueprint(evaluation_views.bp)
    app.register_blueprint(main_views.bp)

    return app