import os

from simple_app_optional.app import app


if __name__ == "__main__":
    app.run(debug=False, port=int(os.environ.get("PORT", "5000")))
