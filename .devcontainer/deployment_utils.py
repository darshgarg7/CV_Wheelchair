import os
import tensorflow as tf
from datetime import datetime
import boto3
import logging
from botocore.exceptions import NoCredentialsError, ClientError

# Configure logging for the application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def save_model_for_serving(model, model_name, version=None, storage_path='models', s3_bucket=None):
    """
    Saves the trained model in a format suitable for serving in production environments.
    Designed for CI/CD workflows with model versioning and scalable storage options.
    """
    version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
    model_save_path = os.path.join(storage_path, model_name, version)
    
    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        model.save(model_save_path, save_format='tf')  # Save the model in TensorFlow's format
        logger.info(f"Model {model_name} version {version} saved successfully at {model_save_path}")

        if s3_bucket:
            upload_model_to_s3(model_save_path, model_name, version, s3_bucket)
        
    except Exception as e:
        logger.error(f"Error saving model {model_name} version {version}: {e}")
        raise

    try:
        logger.info(f"Deployment of model {model_name} version {version} tracked successfully.")
    except Exception as e:
        logger.error(f"Error logging deployment: {e}")
        raise

def upload_model_to_s3(local_model_path, model_name, version, s3_bucket):
    """
    Uploads the model to AWS S3 for centralized storage and version tracking.
    """
    try:
        s3_client = boto3.client('s3')
        s3_model_path = f'{model_name}/{version}/'
        
        # Walk through the local model path and upload files to S3
        for root, files in os.walk(local_model_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(local_file_path, local_model_path)
                s3_file_path = os.path.join(s3_model_path, relative_path)
                
                s3_client.upload_file(local_file_path, s3_bucket, s3_file_path)
                logger.info(f"Uploaded {local_file_path} to s3://{s3_bucket}/{s3_file_path}")
        
        logger.info(f"Model {model_name} version {version} uploaded to S3 successfully.")
    
    except NoCredentialsError:
        logger.error("AWS credentials not found. Please ensure that your AWS credentials are configured.")
        raise
    except ClientError as e:
        logger.error(f"Error uploading to S3: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during S3 upload: {e}")
        raise
