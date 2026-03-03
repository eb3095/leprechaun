"""Flask web UI view handlers for Leprechaun trading bot."""

from flask import Blueprint, render_template, redirect, url_for, flash, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity, verify_jwt_in_request
from functools import wraps

web_bp = Blueprint('web', __name__, 
                   template_folder='templates',
                   static_folder='static',
                   static_url_path='/web/static')


def jwt_optional_web(f):
    """Decorator to optionally verify JWT for web routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request(optional=True)
        except Exception:
            pass
        return f(*args, **kwargs)
    return decorated_function


def get_current_user():
    """Get current user identity if authenticated."""
    try:
        verify_jwt_in_request(optional=True)
        return get_jwt_identity()
    except Exception:
        return None


@web_bp.route('/')
def index():
    """Redirect to dashboard or login based on auth state."""
    user = get_current_user()
    if user:
        return redirect(url_for('web.dashboard'))
    return redirect(url_for('web.login'))


@web_bp.route('/login')
def login():
    """Render login page."""
    user = get_current_user()
    if user:
        return redirect(url_for('web.dashboard'))
    return render_template('login.html')


@web_bp.route('/dashboard')
@jwt_optional_web
def dashboard():
    """Main dashboard view with portfolio overview."""
    return render_template('dashboard.html', 
                          page_title='Dashboard',
                          active_page='dashboard')


@web_bp.route('/positions')
@jwt_optional_web
def positions():
    """Positions view with current holdings."""
    return render_template('positions.html',
                          page_title='Positions',
                          active_page='positions')


@web_bp.route('/trades')
@jwt_optional_web
def trades():
    """Trade history view with filtering and pagination."""
    return render_template('trades.html',
                          page_title='Trade History',
                          active_page='trades')


@web_bp.route('/sentiment')
@jwt_optional_web
def sentiment():
    """Sentiment monitor view."""
    return render_template('sentiment.html',
                          page_title='Sentiment Monitor',
                          active_page='sentiment')


@web_bp.route('/settings')
@jwt_optional_web
def settings():
    """Settings and configuration view."""
    return render_template('settings.html',
                          page_title='Settings',
                          active_page='settings')


@web_bp.route('/alerts')
@jwt_optional_web
def alerts():
    """Alerts history view."""
    return render_template('alerts.html',
                          page_title='Alerts',
                          active_page='alerts')


@web_bp.route('/logout')
def logout():
    """Handle logout (client-side token removal)."""
    return redirect(url_for('web.login'))
