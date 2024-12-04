import ast
import os

class FlaskRouteParser(ast.NodeVisitor):
    def __init__(self, filepath):
        self.routes = []
        self.filepath = filepath

    def visit_FunctionDef(self, node):
        route = None
        template = None

        # Check for Flask @app.route decorator
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call) and hasattr(decorator.func, 'attr') and decorator.func.attr == 'route':
                route = decorator.args[0].s  # Get route URL (first argument in @app.route)

        # Check for render_template call inside the function
        for n in ast.walk(node):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'render_template':
                template = n.args[0].s  # Get the HTML file name passed to render_template

        # Append the route with or without a template
        if route:
            self.routes.append((route, template))  # Template will be None if not found

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
        print("Route Declarations and Associated HTML Files:")
        for route, template in routes:
            if template:
                print(f"Route: {route}, HTML File: {template}")
            else:
                print(f"Route: {route}, HTML File: None")
    else:
        print("No routes found.")
else:
    print(f"File '{flask_app_file}' not found.")
