# import requirements needed
from flask import Flask, render_template, request, redirect, url_for, session, flash
from utils import get_base_url
from video_writer import *
import os, shutil
import time
from werkzeug.utils import secure_filename

# setup the webserver
# port may need to be changed if there are multiple flask servers running on same server
port = 5000
base_url = get_base_url(port)

# if the base url is not empty, then the server is running in development, and we need to specify the static folder so that the static files are served
if base_url == '/':
    app = Flask(__name__)
else:
    app = Flask(__name__, static_url_path=base_url+'static')

writer = VideoWriter()

# initialize backend to handle uploaded files
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
# adds upload folder to base app directory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1000 * 1000
app.secret_key = os.urandom(64)

# tests if file is a valid extension
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#clears uploads folder on flask app run
for filename in os.listdir(UPLOAD_FOLDER):
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
      
# set up the routes and logic for the webserver
@app.route(f'{base_url}', methods=["POST","GET"])
def home():
    
    if request.method == "GET":
        return render_template('index.html')
    elif request.method == "POST":
        if 'file' not in request.files:
            flash('no file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash("no selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            # sanitizes and locally saves the image while the session is running
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            avg, problem_frames, scores = writer.write()
            print("Average score: " + str(round(avg, 2)))
            print("Problem frames: " + str(problem_frames))
            # print("Scores: " + str(scores))
            for score in scores:
                #trim scores to 2 decimal places
                score = round(score, 3)
            time.sleep(len(scores) / 50)
            session['filename'] = filename.split(".")[0]
            session['avg'] = avg
            session['problem_frames'] = problem_frames
            session['scores'] = scores.tolist()
            return redirect(url_for('results', filename='static/uploads/' + session['filename'] + "_w.mp4", score=session['avg'], frames=session['problem_frames'], scores=session['scores']), code=301)

        # add model call here!
    else:
        print("You broke it")
        return redirect(url_for('home'))

@app.route(f"{base_url}/results")
def results():
    global metric_graph
    return render_template('results.html', filename='static/uploads/' + session['filename'] + "_w.mp4", score=session['avg'], frames=session['problem_frames'], scores=session['scores'])

@app.route(f'{base_url}/<filename>')
def display_video(filename):
	#print('display_video filename: ' + filename)
	return redirect(url_for('static', filename= '/uploads/' + filename), code=301)


# define additional routes here
# for example:
# @app.route(f'{base_url}/team_members')
# def team_members():
#     return render_template('team_members.html') # would need to actually make this page

if __name__ == '__main__':
    # IMPORTANT: change url to the site where you are editing this file.
    website_url = 'url'
    
    print(f'Try to open\n\n    https://{website_url}' + base_url + '\n\n')
    app.run(host = '0.0.0.0', port=port, debug=True)


