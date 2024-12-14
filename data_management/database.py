import sqlite3
import json
import logging
from contextlib import closing

# Initialize logging for detailed output and tracking
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def apply_pragma_settings(conn):
    """
    Apply various PRAGMA settings for SQLite optimization.
    """
    PRAGMA_SETTINGS = {
        'foreign_keys': 'ON',  # Ensure foreign key checks are enabled
        'journal_mode': 'WAL',  # Use Write-Ahead Logging for better concurrency
        'synchronous': 'NORMAL',  # Set synchronous mode to NORMAL for better performance
        'cache_size': 10000,  # Adjust cache size for performance tuning
    }

    for key, value in PRAGMA_SETTINGS.items():
        conn.execute(f"PRAGMA {key} = {value};")

class DatabaseManager:
    """
    Manages an SQLite database for storing gesture data and logs.
    Includes table creation, CRUD operations, and logging functionality.
    """

    def __init__(self, db_name='gestures.db'):
        """
        Initializes the DatabaseManager with a specific SQLite database file.
        Applies PRAGMA settings for optimization.
        """
        self.db_name = db_name
        self._initialize_database()

    def _initialize_database(self):
        """
        Initializes the database by creating necessary tables and indexes.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                apply_pragma_settings(conn)  # Apply PRAGMA settings on connection
                queries = [
                    '''
                    CREATE TABLE IF NOT EXISTS gestures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gesture_name TEXT UNIQUE NOT NULL,
                        command TEXT NOT NULL,
                        coordinates JSON NOT NULL
                    )
                    ''',
                    '''
                    CREATE TABLE IF NOT EXISTS logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        gesture_id INTEGER,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (gesture_id) REFERENCES gestures(id) ON DELETE CASCADE
                    )
                    ''',
                    'CREATE INDEX IF NOT EXISTS idx_gesture_name ON gestures(gesture_name)',
                    'CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)'
                ]
                for query in queries:
                    cursor.execute(query)
                conn.commit()
                logger.info("Database setup complete with tables and indexes.")
        except sqlite3.Error as e:
            logger.error(f"Error during database setup: {e}")
            raise

    def add_gesture(self, gesture_name, command, coordinates):
        """
        Inserts a new gesture into the database, ensuring valid coordinates.
        """
        if not isinstance(coordinates, list):
            logger.error("Coordinates must be a list.")
            raise ValueError("Coordinates must be a list.")
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                apply_pragma_settings(conn)  # Apply PRAGMA settings on connection
                cursor.execute(
                    '''
                    INSERT INTO gestures (gesture_name, command, coordinates) 
                    VALUES (?, ?, ?)
                    ''', (gesture_name, command, json.dumps(coordinates))
                )
                conn.commit()
                logger.info(f"Gesture '{gesture_name}' added successfully.")
        except sqlite3.IntegrityError:
            logger.warning(f"Gesture '{gesture_name}' already exists.")
        except sqlite3.Error as e:
            logger.error(f"Error adding gesture: {e}")
            raise

    def add_log(self, gesture_id):
        """
        Records the usage of a gesture in the logs table.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                apply_pragma_settings(conn)  # Apply PRAGMA settings on connection
                cursor.execute(
                    '''
                    INSERT INTO logs (gesture_id) VALUES (?)
                    ''', (gesture_id,)
                )
                conn.commit()
                logger.info(f"Log entry added for gesture ID {gesture_id}.")
        except sqlite3.Error as e:
            logger.error(f"Error adding log entry: {e}")
            raise

    def fetch_all_gestures(self):
        """
        Retrieves all gestures from the database.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                apply_pragma_settings(conn)  # Apply PRAGMA settings on connection
                cursor.execute('SELECT * FROM gestures')
                results = cursor.fetchall()
                logger.info(f"Fetched {len(results)} gestures.")
                return results
        except sqlite3.Error as e:
            logger.error(f"Error fetching gestures: {e}")
            return []

    def fetch_gesture_by_name(self, gesture_name):
        """
        Retrieves a gesture by its name.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                apply_pragma_settings(conn)  # Apply PRAGMA settings on connection
                cursor.execute('SELECT * FROM gestures WHERE gesture_name = ?', (gesture_name,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"Gesture '{gesture_name}' retrieved successfully.")
                else:
                    logger.warning(f"Gesture '{gesture_name}' not found.")
                return result
        except sqlite3.Error as e:
            logger.error(f"Error fetching gesture by name: {e}")
            return None
