import ast
import os

class FlaskRouteParser(ast.NodeVisitor):
    def __init__(self, filepath):
        self.routes = []
        self.filepath = filepath

    def visit_FunctionDef(self, node):
        route = None
        functions_called = []

        # Check for Flask @app.route decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr') and decorator.func.attr == 'route':
                route = decorator.args[0].s  # Get route URL (first argument in @app.route)

        # Get all function calls inside the route handler
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
                functions_called.append(n.func.id)  # Get function name

        if route and functions_called:
            self.routes.append((route, functions_called))

    def parse(self):
        with open(self.filepath, "r") as file:
            tree = ast.parse(file.read())
            self.visit(tree)

        return self.routes

# Point this to the file where your Flask routes are defined
flask_app_file = "app.py"  # Change this to the path of your Flask app file

if os.path.exists(flask_app_file):
    parser = FlaskRouteParser(flask_app_file)
    routes = parser.parse()

    if routes:
        print("Route Declarations and Called Functions:")
        for route, functions in routes:
            print(f"Route: {route}, Functions Called: {functions}")
    else:
        print("No routes with function calls found.")
else:
    print(f"File '{flask_app_file}' not found.")
