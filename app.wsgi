import sys
import os

# Add your project directory to the Python path
project_dir = '/home/arjhun/ML-as-a-Web-Service'
if project_dir not in sys.path:
    sys.path.insert(0, project_dir)

# Set environment variables
os.environ['FLASK_ENV'] = 'production'

# Import your Flask app
from app import app as application

print("WSGI file loaded successfully")
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))