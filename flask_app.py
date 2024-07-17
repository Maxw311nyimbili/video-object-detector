from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
from models.inception_v3 import InceptionV3Model  # Make sure this path is correct
from forms import UploadForm  # Make sure this path is correct

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['FRAME_FOLDER'] = 'static/frames'
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200MB limit

model = InceptionV3Model()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'mp4', 'avi', 'mov'}


def extract_frames(video_path, frame_folder):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_filenames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = f'frame_{frame_count:04d}.jpg'
        frame_filepath = os.path.join(frame_folder, frame_filename)
        cv2.imwrite(frame_filepath, frame)
        frame_filenames.append(frame_filename)
        frame_count += 1
    cap.release()
    return frame_filenames


@app.route('/', methods=['GET', 'POST'])
def index():
    form = UploadForm()
    if form.validate_on_submit():
        file = form.video.data
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])
            if not os.path.exists(app.config['FRAME_FOLDER']):
                os.makedirs(app.config['FRAME_FOLDER'])

            file.save(filepath)

            # Clear old frames
            for f in os.listdir(app.config['FRAME_FOLDER']):
                os.remove(os.path.join(app.config['FRAME_FOLDER'], f))

            frame_filenames = extract_frames(filepath, app.config['FRAME_FOLDER'])

            frames = [cv2.imread(os.path.join(app.config['FRAME_FOLDER'], f)) for f in frame_filenames]
            predictions = model.predict(frames)

            # Filter frames with the requested object
            search_query = form.search_query.data
            results = model.search_object(predictions, search_query)
            relevant_frames = [frame_filenames[i] for i, _ in results]

            if relevant_frames:
                return render_template('results.html', results=results, search_query=search_query, frame_filenames=relevant_frames)
            else:
                flash('Object doesn\'t exist!!!')
        else:
            flash('Invalid file type or no file uploaded.')
    return render_template('index.html', form=form)


if __name__ == '__main__':
    app.run(debug=True)
