"""Entry point:  python wsgi.py   |   flask --app wsgi run

Pending database migrations are applied on startup (set AUTO_MIGRATE=0 to disable and
run `flask --app wsgi db upgrade` yourself)."""
import os

from flask_migrate import upgrade

from app import create_app

app = create_app()

if os.environ.get("AUTO_MIGRATE", "1") == "1":
    with app.app_context():
        upgrade(directory=os.path.join(os.path.dirname(__file__), "migrations"))

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
