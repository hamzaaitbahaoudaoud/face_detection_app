import os
from flask import Flask, render_template, request, redirect, url_for
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Assurez-vous que le dossier 'uploads' existe
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'image' not in request.files:
        return redirect(request.url)
    file = request.files['image']
    if file.filename == '':
        return redirect(request.url)

    # Enregistrer l'image téléchargée
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    # Détection de visages
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Dessiner des rectangles autour des visages détectés
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Sauvegarder l'image avec les visages détectés
    output_file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_' + file.filename)
    cv2.imwrite(output_file_path, image)

    return redirect(url_for('show_result', filename='detected_' + file.filename))

@app.route('/result/<filename>')
def show_result(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
