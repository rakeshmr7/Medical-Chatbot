# Counting the number of non-empty lines in the provided HTML code for the home page
home_page_code = """from flask import Flask, render_template, request, redirect, url_for, flash, session
import sqlite3

from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
from sentence_transformers import SentenceTransformer
from src.helper import mycall
import datetime
from werkzeug.utils import secure_filename
from ocr import extract_text_from_image_or_pdf

app = Flask(__name__)
app.secret_key = "your_secret_key"


# Initialize SQLite database
def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS user (
        name TEXT NOT NULL,
        age INTEGER NOT NULL,
        phone TEXT NOT NULL,
        email TEXT NOT NULL PRIMARY KEY,
        password TEXT NOT NULL
    )''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS upload (
    email TEXT NOT NULL PRIMARY KEY,
    filename TEXT NOT NULL,
    filepath TEXT NOT NULL,
    upload_date DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (email) REFERENCES user(email)
    )''')
    conn.commit()
    conn.close()

# Launch page
@app.route('/')
def launch_page():
    return render_template('index.html')

# Login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM user WHERE email = ? AND password = ?', (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['user_email'] = email  # Store email in session
            flash('Login successful!', 'success')
            return redirect(url_for('home'))  # Redirect to home page after successful login
        else:
            flash('Invalid credentials, please try again.', 'error')
    return render_template('login.html')

# Sign-up page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        age = request.form['age']
        phone = request.form['phone']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password == confirm_password:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('INSERT INTO user (name, age, phone, email, password) VALUES (?, ?, ?, ?, ?)', 
                           (name, age, phone, email, password))
            conn.commit()
            conn.close()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match. Please try again.', 'error')

    return render_template('signup.html')

# Home page after login
@app.route('/home')
def home():
    if 'user_email' in session:
        return render_template('home.html')
    else:
        flash('Please log in to access the home page.', 'error')
        return redirect(url_for('login'))

# Profile Page
@app.route('/profile')
def profile():
    if 'user_email' in session:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute('SELECT name, age, phone, email FROM user WHERE email = ?', (session['user_email'],))
        user = cursor.fetchone()
        conn.close()

        if user:
            # Convert tuple to dictionary
            user_data = {
                'name': user[0],
                'age': user[1],
                'phone': user[2],
                'email': user[3]
            }
            print(user_data)
            return render_template('profile.html', user=user_data)  # Pass user details to template
        else:
            flash('User not found.', 'error')
            return redirect(url_for('home'))
    else:
        flash('Please log in to access your profile.', 'error')
        return redirect(url_for('login'))

# Edit Profile Page
@app.route('/edit-profile', methods=['GET', 'POST'])
def edit_profile():
    if 'user_email' in session:
        if request.method == 'POST':
            name = request.form['name']
            age = request.form['age']
            phone = request.form['phone']
            email = session['user_email']  # Email is not editable

            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('UPDATE user SET name = ?, age = ?, phone = ? WHERE email = ?',
                           (name, age, phone, email))
            conn.commit()
            conn.close()

            flash('Profile updated successfully!', 'success')
            return redirect(url_for('profile'))
        else:
            conn = sqlite3.connect('users.db')
            cursor = conn.cursor()
            cursor.execute('SELECT name, age, phone, email FROM user WHERE email = ?', (session['user_email'],))
            user = cursor.fetchone()
            conn.close()

            if user:
                return render_template('edit_profile.html', user=user)
            else:
                flash('User not found.', 'error')
                return redirect(url_for('profile'))
    else:
        flash('Please log in to edit your profile.', 'error')
        return redirect(url_for('login'))

# Logout Route
@app.route('/logout')
def logout():
    session.pop('user_email', None)
    flash('You have been logged out.', 'success')
    return redirect(url_for('login'))

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}

def get_db_connection():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def allowed_file(filename):
          return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload Medical Report Page
@app.route('/upload-report')
def upload_report():
    if 'user_email' in session:
        return render_template('upload_report.html')
    else:
        flash('Please log in to upload reports.', 'error')
        return redirect(url_for('login'))

upload_dir = "UPLOAD_FOLDER"
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
  if 'user_email' in session:
    if request.method == 'POST':
        print("POST request received")  # Debugging step

        # Check if the file is in the request
        if 'medical_report' not in request.files:
            print("No file part in the request")  # Debugging step
            return 'No file part'
        
        file = request.files['medical_report']  # Match the name attribute in HTML form
        
        if file.filename == '':
            print("No file selected")  # Debugging step
            return 'No selected file'
        
        # Secure the filename
        filename = secure_filename(file.filename)
        print(f"Filename: {filename}")  # Debugging step

        # Store the filename in the session
        session['uploaded_file'] = filename

        # Define the file path for saving the uploaded file
        filepath = os.path.join(upload_dir, filename)

        # Save the file to the upload folder
        file.save(filepath)
        
        # Insert file details into the upload table
        conn = get_db_connection()
        conn.execute("INSERT INTO upload (email, filename, filepath) VALUES (?, ?, ?)",
                     (session['user_email'], filename, filepath))
        conn.commit()
        conn.close()
        
        return jsonify({"message": "File uploaded successfully1"})

    return jsonify({"message": "File uploaded successfully2"})

    
@app.route('/analyze', methods=['GET', 'POST'])
def analyze_report():
 

        if 'uploaded_file' not in session:
          return jsonify({"summary": "No report uploaded yet1"})
        else:
            print("report uploaded1")
        
        filename = session['uploaded_file']
        if not filename:
         return jsonify({"summary": "No report uploaded yet2"})
        else:
            print("report uploaded2")

        filepath = os.path.join(upload_dir, filename)
        extracted_text = extract_text_from_image_or_pdf(filepath)
        summary = answer_question(extracted_text)
    
        return jsonify({"summary": summary})

# Route to display saved reports
@app.route('/reports')
def my_reports():
    conn = get_db_connection()
    email = session['user_email']
    files = conn.execute("SELECT * FROM upload WHERE email = ?", (email,)).fetchall()
    conn.close()
    return render_template('my_report.html', files=files)

# Medical Tests Info Page
@app.route('/medical-tests')
def medical_tests():
    if 'user_email' in session:
        return render_template('medical_tests.html')
    else:
        flash('Please log in to access medical tests information.', 'error')
        return redirect(url_for('login'))
    

load_dotenv()

PINECONE_API_KEY = "7ad73200-8c0f-4d58-8dc5-0731ed6224ed"
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

# Create an instance of the Pinecone class
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if the index exists, if not, create it
index_name = "medical-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Match this to the embedding model output size
        metric='cosine',  # You can use 'cosine', 'euclidean', or 'dotproduct'
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust based on your region
    )

# Now get the index instance
index = pc.Index(index_name)

# Load the SentenceTransformer model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = SentenceTransformer(model_name)

# Function to retrieve from Pinecone
def retrieve_from_pinecone(query_text, top_k=2):
    query_embedding = embeddings.encode(query_text)
    query_embedding_list = query_embedding.tolist()
    
    response = index.query(vector=query_embedding_list, top_k=top_k)
    return response['matches']

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

# Initialize the language model
llm = CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens': 512, 'temperature': 0.8})

def answer_question(question):
    matches = retrieve_from_pinecone(question)
    # Create a context based on IDs if metadata is not available
    context = "\n".join([f"ID: {match['id']}, Score: {match['score']}" for match in matches])
    prompt = PROMPT.format(context=context, question=question)
    return llm.invoke(prompt)

# MChatbot Page
@app.route('/chatbot')
def mchatbot():
    if 'user_email' in session:
        return render_template('chatbot.html')
    else:
        flash('Please log in to access MChatbot.', 'error')
        return redirect(url_for('login'))
    
@app.route("/get", methods=["GET", "POST"])
def chatbot():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = answer_question(input)
    print("Response", result)

    
    return result


# Initialize the database and run the app
if __name__ == '__main__':
    init_db()
    app.run(host="0.0.0.0", port= 8080, debug= True)
"""

# Count the number of non-empty lines in the home page code
home_page_non_empty_lines = [line for line in home_page_code.splitlines() if line.strip()]
num_home_page_non_empty_lines = len(home_page_non_empty_lines)
print(num_home_page_non_empty_lines)
