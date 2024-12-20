Email Spam Classification Using NLP and Machine Learning
This project is a Machine Learning application developed using Natural Language Processing (NLP) and Machine Learning techniques to classify emails as Spam or Not Spam (Ham). The application provides an interactive web interface using Streamlit for real-time email classification.

Table of Contents
Overview
Technologies Used
Features
Setup and Installation
How to Use
Project Demo
Directory Structure
Future Enhancements
Challenges Faced
Acknowledgements
License
Overview
This project demonstrates how NLP techniques such as text vectorization (using CountVectorizer) and machine learning models can be used to classify emails as spam or ham. The application is deployed locally using Streamlit, allowing users to input email text and classify it on the fly.

Technologies Used
Programming Language: Python 3.8+
Libraries:
Streamlit: For creating the web interface.
scikit-learn: For text vectorization and machine learning model.
pickle: For saving and loading pre-trained models and vectorizers.
pandas: For data manipulation and analysis.
Machine Learning Model: Trained using CountVectorizer and Multinomial Naive Bayes.
Features
Email Classification: Enter email text to check whether it is spam or ham.
Interactive Web Interface: Built using Streamlit for user-friendly interaction.
Pre-trained Model: Leverages a pre-trained model for fast predictions.
Real-time Feedback: Displays results instantly.
Setup and Installation
Follow these steps to run the project locally:

1. Clone the Repository
git clone https://github.com/AbdulSarban/P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning.git
cd P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning
2. Create a Virtual Environment
Windows

python -m venv venv
venv\Scripts\activate
macOS/Linux

python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
pip install -r requirements.txt
4. Train the Model (Optional)
To train the model yourself, run:

python train_model.py
5. Run the Application
Start the Streamlit app:

streamlit run spamDetector.py
6. Open the Web App
Visit the URL provided in the terminal (e.g., http://localhost:8501) to interact with the application.

How to Use
Launch the Streamlit app.
Enter email text in the provided input box.
Click the "Classify" button to classify the email.
View the result displayed as "Spam" or "Ham."
Project Demo
Screenshot 2024-12-07 011839

Model Training Details
Text Preprocessing
Tokenization and removal of punctuation.
Conversion to lowercase.
Stopword removal.
Vectorization
Used CountVectorizer for bag-of-words representation.
Model
Trained using Multinomial Naive Bayes for high performance on text classification tasks.
Performance Metrics
Achieved an accuracy of ~98% on the test dataset.
Directory Structure
P3-Spam-Email-Classification-Using-NLP-and-Machine-Learning/
│
├── spamDetector.py             # Main application script
├── train_model.py              # Script for training the model
├── spam.pkl                    # Pre-trained machine learning model
├── vectorizer.pkl              # Pre-trained CountVectorizer
├── requirements.txt            # List of dependencies
├── README.md                   # Project documentation
├── screenshot.png              # Screenshot of the web interface
└── spam.csv                    # Dataset used for training
Future Enhancements
Deployment: Host the app on platforms like Heroku or AWS for broader accessibility.
Multilingual Support: Extend the classification to handle multiple languages.
Advanced Models: Incorporate deep learning models for improved accuracy.
Explainability: Add visualizations to explain model decisions.
User Authentication: Implement user logins to save classification history.
Challenges Faced
Model Overfitting: Resolved using cross-validation and hyperparameter tuning.
Text Preprocessing: Dealt with removing noisy data and handling special characters.
Real-time Predictions: Ensured low latency while maintaining accuracy.
Acknowledgements
Scikit-learn Documentation
Streamlit Tutorials
