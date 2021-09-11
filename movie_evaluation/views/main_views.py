from flask import Blueprint, url_for
from werkzeug.utils import redirect

bp = Blueprint('main', __name__, url_prefix='/')

@bp.route('/')
def redirectToMain():
    return redirect(url_for('evaluation.page'))