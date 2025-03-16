import uuid
from flask import Flask, request, render_template, jsonify, url_for
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload and report folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('static/reports', exist_ok=True)

# Load the YOLOv8 model (update 'best.pt' with your model's path)
model = YOLO('best.pt')  # Ensure this path is correct

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# PDF generation function
def generate_pdf(unique_filename, label, confidence):
    """
    Generates a PDF report for the prediction.
    The PDF is saved in 'static/reports' with the same base name as the uploaded image.
    """
    pdf_filename = f"{unique_filename}.pdf"
    pdf_filepath = os.path.join('static', 'reports', pdf_filename)
    
    doc = SimpleDocTemplate(pdf_filepath, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Add title and prediction details
    story.append(Paragraph("Flood Damage Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Prediction: {label}", styles['Normal']))
    story.append(Paragraph(f"Confidence: {confidence * 100:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Add the image thumbnail
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    try:
        story.append(Image(image_path, width=200, height=200))
    except Exception as e:
        story.append(Paragraph("Error loading image.", styles['Normal']))
    
    doc.build(story)
    return pdf_filename

@app.route('/')
def reg():
    return render_template('reg.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/pesticide')
def pesticide():
    return render_template('pesticide_details.html')

@app.route('/fertilizers')
def fertilizers():
    return render_template('fertilizers.html')

@app.route('/schemes')
def schemes():
    return render_template('schemes.html')

@app.route('/about')
def about():
    return render_template('about.html')

# Handle the image upload and YOLO prediction
@app.route('/upload', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        # Check if the file is in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        
        if file and allowed_file(file.filename):
            # Save the uploaded file with a unique filename
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Perform YOLO model prediction
            try:
                results = model.predict(filepath)
                if results and len(results) > 0:
                    probs = results[0].probs
                    if probs is not None:
                        label_index = probs.top1
                        confidence = probs.top1conf.item()
                        label = results[0].names[label_index]
                    else:
                        label = "No prediction"
                        confidence = 0.0
                else:
                    label = "No prediction"
                    confidence = 0.0
                
                # Generate PDF report with the prediction
                pdf_filename = generate_pdf(unique_filename, label, confidence)
                
                # Render result page with prediction and PDF download link
                return render_template(
                    'result.html',
                    label=label,
                    confidence=f"{confidence:.2%}",
                    image_url=url_for('static', filename=f'uploads/{unique_filename}'),
                    pdf_url=url_for('static', filename=f'reports/{pdf_filename}')
                )
            except Exception as e:
                return jsonify({"error": f"Prediction failed: {str(e)}"})
    
    # If it's a GET request, show the upload form
    return render_template('index.html')

@app.route('/upload_progress', methods=['GET', 'POST'])
def upload_progress():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file and allowed_file(file.filename):
            unique_filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_PROGRESS_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Save progress upload info (for crop progress, we may not need a prediction or PDF)
            progress_history.append({
                "filename": unique_filename,
                "image_url": url_for('static', filename=f'uploads_progress/{unique_filename}')
            })
            return redirect(url_for('history_progress'))
    return render_template('upload_progress.html')

@app.route('/history_progress')
def history_progress():
    return render_template('history_progress.html', history=progress_history)

# Starting the app
if __name__ == '__main__':
    app.run(debug=True)
