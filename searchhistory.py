import sqlite3
import os
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

def mk_temp(data):  # Create a temporary file
    with open('tempfile.txt', 'a') as inputs:
        inputs.write(data + '\n')  # Ensure each entry is on a new line

# Create the database and table if they don't exist
def create_database():
    conn = sqlite3.connect('history.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS history (command TEXT PRIMARY KEY)')  # Set command as PRIMARY KEY

    # Path to the Bash history file
    history_file = os.path.expanduser('~/.bash_history')

    # Read the history file and insert commands into the database
    if os.path.exists(history_file):
        with open(history_file, 'r') as file:
            for line in file:
                command = line.strip()  # Remove leading/trailing whitespace
                if command:  # Check if the line is not empty
                    try:
                        cursor.execute('INSERT INTO history (command) VALUES (?)', (command,))
                    except sqlite3.IntegrityError:
                        # This error occurs if the command already exists
                        pass

    conn.commit()
    conn.close()

# Call create_database() to initialize the database when the app starts
create_database()

# Function to get a database connection
def get_db_connection():
    conn = sqlite3.connect('history.db')
    conn.row_factory = sqlite3.Row
    return conn

# Load the index page
@app.route('/', methods=['GET', 'POST'])
def index():
    history_results = []  # Initialize results for command history

    # Check for command history search
    if request.method == 'POST' and 'query' in request.form:
        query = request.form['query']
        with get_db_connection() as conn:
            cursor = conn.execute('SELECT command FROM history WHERE command LIKE ?', ('%' + query + '%',))
            history_results = cursor.fetchall()  # Fetch all matching results from the database

        # Create a temporary file for found results
        for data in history_results:
            mk_temp(data[0])  # Create a temporary file for each found result

    return render_template('search_history.html', history_results=history_results)

# Load the search for the tempfile
@app.route('/search_text_file', methods=['POST'])
def search_text_file():
    temp_results = []  # Initialize results for tempfile.txt

    # Check for tempfile search
    if 'temp_query' in request.form:
        temp_query = request.form['temp_query']
        if os.path.exists('tempfile.txt'):
            with open('tempfile.txt', 'r') as temp_file:
                temp_results = [line.strip() for line in temp_file if temp_query in line]  # Search for temp_query in tempfile.txt

    return render_template('search_history.html', temp_results=temp_results)

# Route to remove tempfile.txt
@app.route('/remove_tempfile', methods=['POST'])
def remove_tempfile():
    if os.path.exists('tempfile.txt'):
        os.remove('tempfile.txt')  # Remove the tempfile
    return redirect(url_for('index'))  # Redirect back to the index page

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5200)


