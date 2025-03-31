import sqlite3
import json
import logging
from contextlib import closing

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


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
                queries = [
                    '''
                    CREATE TABLE IF NOT EXISTS gestures (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL UNIQUE,
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
                    'CREATE INDEX IF NOT EXISTS idx_gesture_name ON gestures(name)',
                    'CREATE INDEX IF NOT EXISTS idx_timestamp ON logs(timestamp)'
                ]
                for query in queries:
                    cursor.execute(query)
                conn.commit()
                logger.info("Database setup complete with tables and indexes.")
        except sqlite3.Error as e:
            logger.error(f"Error during database setup: {e}")
            raise

    def save_gesture(self, gesture_name: str, command: str, coordinates: dict):
        """
        Saves a recognized gesture into the database.
        If the gesture already exists, updates its command and coordinates.
        Logs the gesture usage in the logs table.
        Args:
            gesture_name (str): Name of the gesture.
            command (str): Command associated with the gesture.
            coordinates (dict): Gesture coordinates stored as a dictionary.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                # Check if the gesture already exists
                cursor.execute('SELECT id FROM gestures WHERE name = ?', (gesture_name,))
                result = cursor.fetchone()

                if result:
                    gesture_id = result[0]
                    logger.info(f"Gesture '{gesture_name}' already exists in the database. Updating...")
                    cursor.execute(
                        '''
                        UPDATE gestures 
                        SET command = ?, coordinates = ? 
                        WHERE id = ?
                        ''', (command, json.dumps(coordinates), gesture_id)
                    )
                else:
                    # Insert the new gesture
                    cursor.execute(
                        '''
                        INSERT INTO gestures (name, command, coordinates) 
                        VALUES (?, ?, ?)
                        ''', (gesture_name, command, json.dumps(coordinates))
                    )
                    gesture_id = cursor.lastrowid
                    logger.info(f"Gesture '{gesture_name}' added successfully.")

                # Log the gesture usage
                cursor.execute(
                    '''
                    INSERT INTO logs (gesture_id) VALUES (?)
                    ''', (gesture_id,)
                )
                conn.commit()
                logger.info(f"Log entry added for gesture ID {gesture_id}.")
        except sqlite3.IntegrityError:
            logger.warning(f"Gesture '{gesture_name}' already exists.")
        except sqlite3.Error as e:
            logger.error(f"Error saving gesture: {e}")
            raise

    def fetch_all_gestures(self):
        """
        Fetches all gestures from the database.
        Returns:
            list: A list of tuples containing gesture data (id, name, command, coordinates).
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                cursor.execute('SELECT * FROM gestures')
                results = cursor.fetchall()
                logger.info(f"Fetched {len(results)} gestures from the database.")
                return results
        except sqlite3.Error as e:
            logger.error(f"Error fetching gestures: {e}")
            return []

    def fetch_gesture_by_name(self, gesture_name: str):
        """
        Fetches a gesture by its name.
        Args:
            gesture_name (str): Name of the gesture to fetch.
        Returns:
            tuple: A tuple containing gesture data (id, name, command, coordinates), or None if not found.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                cursor.execute('SELECT * FROM gestures WHERE name = ?', (gesture_name,))
                result = cursor.fetchone()
                if result:
                    logger.info(f"Gesture '{gesture_name}' retrieved successfully.")
                else:
                    logger.warning(f"Gesture '{gesture_name}' not found.")
                return result
        except sqlite3.Error as e:
            logger.error(f"Error fetching gesture by name: {e}")
            return None

    def update_gesture(self, gesture_name: str, new_command: str, new_coordinates: dict):
        """
        Updates the command and coordinates of an existing gesture.
        Args:
            gesture_name (str): Name of the gesture to update.
            new_command (str): New command for the gesture.
            new_coordinates (dict): New coordinates for the gesture.
        Returns:
            bool: True if the update was successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                cursor.execute(
                    '''
                    UPDATE gestures 
                    SET command = ?, coordinates = ? 
                    WHERE name = ?
                    ''', (new_command, json.dumps(new_coordinates), gesture_name)
                )
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Gesture '{gesture_name}' updated successfully.")
                    return True
                else:
                    logger.warning(f"Gesture '{gesture_name}' not found for update.")
                    return False
        except sqlite3.Error as e:
            logger.error(f"Error updating gesture: {e}")
            return False

    def delete_gesture(self, gesture_name: str):
        """
        Deletes a gesture from the database.
        Args:
            gesture_name (str): Name of the gesture to delete.
        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                cursor.execute('DELETE FROM gestures WHERE name = ?', (gesture_name,))
                if cursor.rowcount > 0:
                    conn.commit()
                    logger.info(f"Gesture '{gesture_name}' deleted successfully.")
                    return True
                else:
                    logger.warning(f"Gesture '{gesture_name}' not found for deletion.")
                    return False
        except sqlite3.Error as e:
            logger.error(f"Error deleting gesture: {e}")
            return False

    def fetch_logs(self):
        """
        Fetches all logs from the database.
        Returns:
            list: A list of tuples containing log data (id, gesture_id, timestamp).
        """
        try:
            with sqlite3.connect(self.db_name) as conn, closing(conn.cursor()) as cursor:
                cursor.execute('SELECT * FROM logs')
                results = cursor.fetchall()
                logger.info(f"Fetched {len(results)} logs from the database.")
                return results
        except sqlite3.Error as e:
            logger.error(f"Error fetching logs: {e}")
            return []
