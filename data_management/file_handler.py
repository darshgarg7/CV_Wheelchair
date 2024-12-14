import json
import logging
import os
import shutil

# Initialize logging configuration to keep track of important events.
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class FileHandler:
    """
    A class to handle file operations for storing and loading gesture data.
    """

    @staticmethod
    def save_gesture_to_file(gesture_data, file_name='gestures.json'):
        """
        Saves gesture data to a JSON file.
        """
        if not isinstance(gesture_data, dict):
            raise ValueError("gesture_data must be a dictionary.")

        try:
            backup_file = f"{file_name}.bak"
            if os.path.exists(file_name):
                shutil.copy(file_name, backup_file)
                logger.info(f"Backup created: {backup_file}")

            with open(file_name + ".tmp", 'w') as tmp_file:
                json.dump(gesture_data, tmp_file, indent=4)
            os.rename(file_name + ".tmp", file_name)
            logger.info(f"Gesture data saved to {file_name}.")
        except Exception as e:
            logger.error(f"Error saving gesture data to file: {e}")

    @staticmethod
    def load_gesture_from_file(file_name='gestures.json'):
        """
        Loads gesture data from a JSON file.
        """
        try:
            with open(file_name, 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            logger.warning(f"File {file_name} not found. Returning empty data.")
            return {}
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {file_name}.")
            return {}
        except Exception as e:
            logger.error(f"Error loading gesture data from file: {e}")
            return {}
