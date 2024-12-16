s# Computer Vision Wheelchair with Gesture Augmentation

This repository implements a state-of-the-art gesture-controlled wheelchair system, combining EfficientNetB7 as a backbone for image classification, GAN-based synthetic data augmentation to handle class imbalances, and an IoT-powered control mechanism.
The system integrates CI/CD pipelines for continuous deployment and is designed to showcase cutting-edge AI and software engineering practices in inclusive design and accessibility; there is a SQLite Database for storing gestures and logs with Django Integration.

## **Key Features**
- **Gesture Recognition and Wheelchair Control**:  
  - **EfficientNetB7 Backbone**: Pretrained on ImageNet and fine-tuned for precise gesture classification. Recognizes gestures mapped to wheelchair controls:  
    - **Fist**: Stop  
    - **One Finger**: Move Forward  
    - **Thumbs Left**: Turn Left  
    - **Thumbs Right**: Turn Right  
- **GAN-Augmented Dataset**:  
  - Custom GAN generates synthetic gesture images to handle class imbalances and enhance model robustness.
- **Training Optimization**:  
  - **Hyperparameter Tuning**: Utilizes Keras Tuner's Hyperband for optimized model performance.  
  - **Callbacks**: Implements **EarlyStopping**, **ModelCheckpoint**, and **TensorBoard** for efficient training and preventing overfitting.
- **Data Management**:  
  - **SQLite Database**: Uses a comprehensive schema to track gestures and actions in real-time.  
  - **Django Integration**: Structured for easy retrieval and updating of gesture data.
- **Deployment & Operations**:  
  - **CI/CD Pipeline**: Ensures scalable, rapid updates with Docker for continuous deployment.  
  - **Logging & Error Handling**: Detailed logs for database operations and model management ensure reliability and transparency.

# **Database and Gesture Management**
The system uses SQLite to manage gesture data efficiently:
Gestures Table: Stores gesture names, commands, and coordinates in JSON format.
Logs Table: Tracks gesture recognition events with timestamps.
Database Management: The database.py file handles CRUD operations for gestures and logs, while file_handler.py supports saving and loading gesture data from files.

## **Project Structure**




## **Installation**
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/darshgarg7/CV_Wheelchair.git
   cd CV_Wheelchair/.devcontainer
2. **Build the Docker Container**:
   docker build -t cv_wheelchair /path/to/.devcontainer  #add path to .devcontainer

## **Usage**
1. python3 gesture_model.py  #Train the Model
2. tensorboard --logdir=./logs  #Monitor Training with TensorBoard
3. python3 -m utils.deployment_utils  #Deploy the Model
4. python3 controller.py  #Run Wheelchair Controller
5. python3 app.py #Run Server
     - curl http://localhost:8000/health #Perform Health Check

# **Testing**
1. python3 -m unittest discover

# **Contributing**
- Please follow the guidelines in the CONTRIBUTING.txt for submitting issues and pull requests.

# **Lisence**
- This project is licensed under the GNU AFFERO GENERAL PUBLIC LICENSE - see the LICENSE file for details.

# **Acknowledgements**
- TensorFlow/Keras for providing the machine learning framework 🍾
- EfficientNet authors for their contribution to model efficiency 🥳
- The SQLite team for their lightweight database management system used in storing gesture data 😇

# **Future Improvements**
- Add voice-command integration for enhanced usability.
- Implement obstacle detection using ultrasonic sensors for safety.
- Scale gesture recognition to support more complex commands.

# **Contact Me 😃**
- Darsh Garg
- Email: darsh.garg@gmail.com
- Phone Number:  +1 (612)819-2636
