from flask import Flask, render_template, request, g, redirect, url_for
import sqlite3

from fetch_trends import fetch_trending_searches, store_trending_searches

app = Flask(__name__)

# Configure logging


# Database configuration
DATABASE = 'trending_searches.db'

def get_db():
    """Establish a database connection."""
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    """Close the database connection at the end of the request."""
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

@app.route('/')
def search_trends():
    """Render the homepage with trending searches."""
    db = get_db()
    cur = db.execute('SELECT term, timestamp FROM trends ORDER BY timestamp DESC')
    trends = cur.fetchall()
    return render_template('search_trends.html', trends=trends)

@app.route('/update_trends')
def update_trends():
    """Fetch and store the latest trending searches."""
    trends = fetch_trending_searches()
    store_trending_searches(trends)
    return redirect(url_for('search_trends'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5100)
