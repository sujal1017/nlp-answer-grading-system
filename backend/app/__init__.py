import logging
import os

from flask import Flask, jsonify
from flask_cors import CORS
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from .config import Config
from .extensions import db, migrate


def create_app(config_object=None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config.from_object(config_object or Config)
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite:///"):
        path = app.config["SQLALCHEMY_DATABASE_URI"].replace("sqlite:///", "", 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    CORS(app, origins=app.config.get("CORS_ORIGINS", []))
    db.init_app(app)
    migrate.init_app(app, db)

    from . import models  # noqa: F401  (register models for migrations)
    from .routes.api import bp

    app.register_blueprint(bp)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_):
        return jsonify({"error": f"file exceeds the {app.config['MAX_UPLOAD_MB']} MB limit"}), 413

    @app.errorhandler(HTTPException)
    def http_error(e):
        return jsonify({"error": e.description}), e.code

    @app.errorhandler(Exception)
    def unexpected(e):
        db.session.rollback()
        app.logger.exception("unhandled error")
        return jsonify({"error": "internal server error"}), 500

    return app
