"""Tiny Flask-style stub app (no real Flask required)."""
from routes import users, memos, health


def create_app():
    app = StubApp()
    app.register(users.routes)
    app.register(memos.routes)
    app.register(health.routes)
    return app


class StubApp:
    def __init__(self):
        self.routes = []

    def register(self, route_list):
        self.routes.extend(route_list)


if __name__ == "__main__":
    app = create_app()
    print(f"App registered {len(app.routes)} routes")
