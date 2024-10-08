from flask import Flask, request, jsonify, redirect, url_for
import jwt
import datetime
from transformers import CLIPProcessor, CLIPModel, BartForSequenceClassification, BartTokenizer
from PIL import Image
import torch
import io
import base64
from flask_cors import CORS  # Import CORS
from werkzeug.middleware.proxy_fix import ProxyFix  # Middleware for HTTPS redirection


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}) 


# Ensure app uses HTTPS (behind reverse proxy)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Secret key for JWT encoding/decoding
SECRET_KEY = 'asis_mobileapp'

@app.before_request
def enforce_https():
    """Redirect HTTP requests to HTTPS"""
    if not request.is_secure:
        return redirect(request.url.replace("http://", "https://"), code=301)
    

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"message": "Missing username or password"}), 400

    try:
        # Verify user's credentials with Firebase
        # user = auth.get_user_by_email(username)

        # Firebase does not allow password verification directly on the backend
        # You should use Firebase's SDK client-side for user authentication and pass a token to the backend

        # For demo purposes, assume user is valid and generate JWT token
        token = jwt.encode({
            'username': username,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)  # Token expires in 1 hour
        }, SECRET_KEY, algorithm='HS256')

        return jsonify({'token': token}), 200
    except Exception as e:
        return jsonify({"message": str(e)}), 500

# Load the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

# Load the BART model and tokenizer for text classification
bart_model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli')
bart_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli')

# Define problem labels and solutions
labels = [
    "Biohazard waste removal", 
    "Floor cleaning and maintenance", 
    "Infection control measures",
    "Plumbing and waste disposal",
    "Elevator malfunction (stuck between floors)",
    "Door sensor not functioning properly",
    "Lift jerking while in motion",
    "Faulty lighting in common areas",
    "Overflowing garbage bins",
    "Mold growth in bathrooms",
    "Malfunctioning air conditioning unit",
    "Water leakage from ceiling"

]
solutions = {
    "Biohazard waste removal": "Ensure timely disposal of biohazard waste.",
    "Floor cleaning and maintenance": "Fix or replace broken floor tiles, or deep clean.",
    "Infection control measures": "Repair or replace faulty infection control equipment.",
    "Plumbing and waste disposal": "Fix or replace broken plumbing fixtures.",
    "Elevator malfunction (stuck between floors)": "Check and reset the elevator motor and control system. Recalibrate sensors, and ensure no mechanical obstructions are present.",
    "Door sensor not functioning properly":"Inspect and adjust the door sensor alignment. Clean any debris blocking the sensor and test the doors for proper operation.",
    "Lift jerking while in motion":"Perform a thorough inspection of the lift motor, cables, and braking system. Replace worn components and test for smooth operation.",
    "Faulty lighting in common areas":"Replace broken or flickering light bulbs and check wiring. Consider upgrading to energy-efficient lighting systems for longevity.",
    "Overflowing garbage bins": "Schedule more frequent garbage collection. Ensure that bins are emptied and cleaned regularly to avoid health hazards.",
    "Mold growth in bathrooms":"Deep clean affected areas with anti-mold cleaning agents. Investigate and fix any leaks or ventilation issues causing excess moisture.",
    "Malfunctioning air conditioning unit":"Inspect and repair or replace faulty components. Clean filters and check refrigerant levels for optimal performance.",
    "Water leakage from ceiling":"Identify the source of the leak, repair any damaged plumbing, and replace affected ceiling tiles. Conduct a thorough inspection to prevent recurrence."


}

problem ="The lift is stuck between the 4th and 5th floors, and the doors are not functioning properly."

@app.route('/classify', methods=['POST'])
def classify_image():
    try:
         # Verify the JWT token from the Authorization header
        token = request.headers.get('Authorization')
        if token:
            token = token.split(" ")[1]  # Bearer token format
            jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        else:
            return jsonify({'error': 'Token is missing!'}), 401
        
        # Get the image from the request
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image file provided.'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1]  # Extract the base64 part after 'data:image/jpeg;base64,'
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess the image and labels
        inputs = processor(text=labels, images=image, return_tensors="pt", padding=True)

        # Perform classification
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: (batch_size, num_labels)
            probs = logits_per_image.softmax(dim=1)  # Convert logits to probabilities

        # Get the label with the highest probability
        predicted_label = labels[probs.argmax().item()]
        solution = solutions.get(predicted_label, "No corresponding solution found.")

        # Return the predicted problem and solution
        return jsonify({
            'predicted_problem': predicted_label,
            'solution': solution
        })  
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired!'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token!'}), 401
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500
    

@app.route('/classify-text', methods=['POST'])
def classify_text():
    try:
         # Verify the JWT token from the Authorization header
        token = request.headers.get('Authorization')
        if token:
            token = token.split(" ")[1]  # Bearer token format
            jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
        else:
            return jsonify({'error': 'Token is missing!'}), 401
        
        # Get the text from the request
        data = request.get_json()
        if 'text' not in data:
            return jsonify({'error': 'No text provided.'}), 400
        
        input_text = data['text']
        # input_text = problem
        # Preprocess the text for BART model
        inputs = bart_tokenizer([input_text] + labels, return_tensors="pt", padding=True, truncation=True)

        # Perform text classification
        with torch.no_grad():
            outputs = bart_model(**inputs)
            logits_per_text = outputs.logits  # Shape: (batch_size, num_labels)
            probs = logits_per_text.softmax(dim=1)  # Convert logits to probabilities

        # Compare the input text with the predefined labels and get the highest probability
        predicted_label = labels[probs[0, 1:].argmax().item()]  
        solution = solutions.get(predicted_label, "No corresponding solution found.")

        # Return the predicted problem and solution
        return jsonify({
            'predicted_problem': predicted_label,
            'solution': solution
        })
    except jwt.ExpiredSignatureError:
        return jsonify({'error': 'Token has expired!'}), 401
    except jwt.InvalidTokenError:
        return jsonify({'error': 'Invalid token!'}), 401
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500    

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000, ssl_context=(
        'myserver.crt',  # Replace with the path to your SSL certificate
        'myserver.key'   # Replace with the path to your SSL key
    ))
