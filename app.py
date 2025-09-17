from flask import Flask, request, render_template, jsonify
from supabase import create_client
import face_recognition
from PIL import Image
import io
import numpy as np
import os
import logging
import json
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv()
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'default_password')

# Initialize Supabase client
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Supabase client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Supabase client: {e}")
    raise


# Load reference embeddings from Supabase
def load_reference_embeddings():
    try:
        response = supabase.table('users').select('id, name, uid, embedding').execute()
        data = response.data
        embeddings = []
        names = []
        for row in data:
            if row['embedding']:
                emb = json.loads(row['embedding'])  # decode JSON string
                embeddings.append(np.array(emb, dtype=np.float64))
                names.append(row['name'])
        logger.info(f"Loaded {len(embeddings)} embeddings from Supabase")
        return embeddings, names
    except Exception as e:
        logger.error(f"Error loading embeddings from Supabase: {e}")
        return [], []


reference_encodings, reference_names = load_reference_embeddings()


# Process uploaded image
def process_image(file):
    file.seek(0)
    img_data = file.read()
    logger.info(f"Raw image data length: {len(img_data)} bytes, type: {type(img_data)}")
    
    try:
        pil_img = Image.open(io.BytesIO(img_data)).convert('RGB')
        logger.info(f"Image mode after conversion: {pil_img.mode}, size: {pil_img.size}")
    except Exception as e:
        logger.error(f"Failed to open image with PIL: {e}")
        raise ValueError("Invalid image format")
    
    img_array = np.array(pil_img, dtype=np.uint8)
    logger.info(f"Image array shape: {img_array.shape}, dtype: {img_array.dtype}")
    
    encodings = face_recognition.face_encodings(img_array)
    if len(encodings) != 1:
        logger.error(f"Face detection failed: {len(encodings)} faces detected")
        raise ValueError(f"{'No face' if len(encodings) == 0 else 'Multiple faces'} detected in image")
    
    return encodings[0]


@app.route('/')
def index():
    return render_template('index.html', reference_names=reference_names)


@app.route('/admin', methods=['POST'])
def admin_login():
    try:
        data = request.get_json()
        password = data.get('password')
        if not password:
            return jsonify({'success': False, 'message': 'Password required'})
        if password == ADMIN_PASSWORD:
            return jsonify({'success': True})
        else:
            return jsonify({'success': False, 'message': 'Invalid password'})
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        return jsonify({'success': False, 'message': 'Error verifying password'})


@app.route('/add_user', methods=['POST'])
def add_user():
    global reference_encodings, reference_names
    logger.info("Received add_user request")
    
    if 'file' not in request.files or not request.form.get('name'):
        return render_template('index.html', message="Image and name are required", reference_names=reference_names)
    
    file = request.files['file']
    name = request.form.get('name').strip()
    uid = request.form.get('uid', '').strip() or None
    
    # Validate file
    try:
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        if file_size > 5 * 1024 * 1024:
            return render_template('index.html', message="Image too large. Max size is 5MB.", reference_names=reference_names)
        file.seek(0)
    except Exception:
        return render_template('index.html', message="Error reading image file", reference_names=reference_names)
    
    try:
        # Process image
        embedding = process_image(file)
        
        # Save to Supabase (as JSON string)
        data = {
            'name': name,
            'uid': uid,
            'embedding': json.dumps(embedding.tolist())
        }
        supabase.table('users').insert(data).execute()
        
        # Refresh reference embeddings
        reference_encodings, reference_names = load_reference_embeddings()
        
        return render_template('index.html', message=f"User {name} added successfully", reference_names=reference_names)
    
    except Exception as e:
        logger.error(f"Error adding user: {str(e)}")
        return render_template('index.html', message=f"Error adding user: {str(e)}", reference_names=reference_names)


@app.route('/verify', methods=['POST'])
def verify():
    logger.info("Received verify request")
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded", reference_names=reference_names)
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No selected file", reference_names=reference_names)
    
    try:
        input_encoding = process_image(file)
        
        if not reference_encodings:
            return render_template('index.html', message="No reference embeddings in database", reference_names=reference_names)
        
        # Check for specific reference
        reference_name = request.form.get('reference_name')
        if reference_name and reference_name in reference_names:
            idx = reference_names.index(reference_name)
            result = face_recognition.compare_faces([reference_encodings[idx]], input_encoding, tolerance=0.4)[0]
            distance = face_recognition.face_distance([reference_encodings[idx]], input_encoding)[0]
            if result:
                return render_template('index.html', message=f"Match! It's {reference_names[idx]} (Similarity: {1 - distance:.2f}).", reference_names=reference_names)
            else:
                return render_template('index.html', message=f"No match with {reference_names[idx]}. Similarity: {1 - distance:.2f}.", reference_names=reference_names)
        
        # Check against all references
        results = face_recognition.compare_faces(reference_encodings, input_encoding, tolerance=0.4)
        distances = face_recognition.face_distance(reference_encodings, input_encoding)
        
        for i, result in enumerate(results):
            if result:
                return render_template('index.html', message=f"Match! It's {reference_names[i]} (Similarity: {1 - distances[i]:.2f}).", reference_names=reference_names)
        
        if distances.size > 0:
            closest_distance = min(distances)
            closest_index = np.argmin(distances)
            return render_template('index.html', message=f"No match. Closest similarity: {1 - closest_distance:.2f} with {reference_names[closest_index]}.", reference_names=reference_names)
        
        return render_template('index.html', message="No match. Different person.", reference_names=reference_names)
    
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return render_template('index.html', message=f"Error processing image: {str(e)}", reference_names=reference_names)


if __name__ == '__main__':
    app.run(debug=False)
