from pytrends.request import TrendReq
import sqlite3


# Configure logging


# Initialize pytrends
pytrends = TrendReq(hl='en-US', tz=360)

# Fetch trending searches in the USA
def fetch_trending_searches():
    trending_searches = pytrends.trending_searches(pn='united_states')
    trends = trending_searches[0].tolist()
    return trends

# Store trending searches in the database
def store_trending_searches(trends):
    conn = sqlite3.connect('trending_searches.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS trends (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            term TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    for term in trends:
        c.execute('INSERT INTO trends (term) VALUES (?)', (term,))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    trends = fetch_trending_searches()
    store_trending_searches(trends)
