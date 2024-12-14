from django.db import models
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

class Gesture(models.Model):
    """
    A Django model to represent a gesture. This model stores the gesture name, associated command, and its coordinates.
    """

    gesture_name = models.CharField(max_length=255, unique=True)
    command = models.CharField(max_length=255)
    coordinates = models.JSONField()

    def save_gesture(self):
        """
        Saves the gesture instance to the database.

        This method ensures the gesture is properly saved and logs any issues encountered during the operation.
        """
        try:
            # Save the instance to the database
            self.save()
            logger.info(f"Gesture '{self.gesture_name}' saved successfully.")
        except Exception as e:
            # Log an error if the save operation fails
            logger.error(f"Error saving gesture '{self.gesture_name}': {e}")

    @classmethod
    def get_gesture_by_name(cls, gesture_name):
        """
        Retrieves a gesture by its name from the database.

        Args:
            gesture_name (str): The name of the gesture to retrieve.

        Returns:
            Gesture: The Gesture object if found, or None if not found.

        Logs any errors encountered during the retrieval process.
        """
        try:
            # Attempt to fetch the gesture by name
            return cls.objects.get(gesture_name=gesture_name)
        except cls.DoesNotExist:
            # If the gesture does not exist, log a warning and return None
            logger.warning(f"Gesture '{gesture_name}' not found.")
            return None
        except Exception as e:
            # Log any other exceptions encountered
            logger.error(f"Error fetching gesture '{gesture_name}': {e}")
            return None

    def __str__(self):
        """
        Returns a string representation of the Gesture object.

        This method is useful for logging and debugging, providing a clear representation of the object.
        """
        return f"Gesture(name={self.gesture_name}, command={self.command})"
