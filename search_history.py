from flask import Flask, request, session, render_template_string
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Needed for session usage

# Function to load and clean terminal history
def load_history():
    # Load the 'history' data from the terminal command
    history_output = os.popen('history').read().splitlines()
    
    # Process history: Remove leading numbers and duplicates
    processed_history = set()  # Use a set to avoid duplicates
    for line in history_output:
        # Remove leading numbers and strip extra spaces
        cleaned_line = ' '.join(line.split()[1:])
        if cleaned_line:
            processed_history.add(cleaned_line)

    return list(processed_history)

# Simple HTML template embedded in the Flask app
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Terminal History Search</title>
</head>
<body>
    <h1>Terminal History Search</h1>
    
    <h2>Load Terminal History</h2>
    <form action="/load_history" method="post">
        <button type="submit">Load History</button>
    </form>

    {% if history_data %}
        <h2>Search History</h2>
        <form action="/search" method="post">
            <input type="text" name="query" placeholder="Enter search query">
            <button type="submit">Search</button>
        </form>

        <h3>Search Results:</h3>
        <ul>
        {% for result in results %}
            <li>{{ result }}</li>
        {% endfor %}
        </ul>
    {% endif %}

    {% if search_history %}
        <h2>Previous Search Queries</h2>
        <ul>
        {% for query in search_history %}
            <li>{{ query }}</li>
        {% endfor %}
        </ul>
    {% endif %}
    
    <form action="/clear_session" method="post">
        <button type="submit">Clear Session</button>
    </form>
</body>
</html>
'''

# Route to load history into memory and display the form
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST' and 'load_history' in request.form:
        # Load and store the history in session
        history_data = load_history()
        session['history_data'] = history_data
    
    # Prepare the data for rendering
    history_data = session.get('history_data', None)
    search_history = session.get('search_history', [])
    results = []

    return render_template_string(HTML_TEMPLATE, history_data=history_data, search_history=search_history, results=results)

# Route to search within the loaded history
@app.route('/search', methods=['POST'])
def search_history():
    query = request.form.get('query', '').lower()
    history_data = session.get('history_data', [])
    results = [item for item in history_data if query in item.lower()]

    # Save the search query to session for repeat searches
    if 'search_history' not in session:
        session['search_history'] = []
    session['search_history'].append(query)

    # Render the page with the search results
    search_history = session.get('search_history', [])
    return render_template_string(HTML_TEMPLATE, history_data=history_data, search_history=search_history, results=results)

# Route to clear session data
@app.route('/clear_session', methods=['POST'])
def clear_session():
    session.clear()
    return render_template_string(HTML_TEMPLATE, history_data=None, search_history=[], results=[])

if __name__ == '__main__':
    app.run(debug=True)
