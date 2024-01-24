# App and Login
from flask import Flask, render_template, url_for, redirect, Response, request, flash, send_file, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import InputRequired, Length, ValidationError, EqualTo, DataRequired
from flask_bcrypt import Bcrypt

# ML
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import os

app=Flask(__name__)
db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SECRET_KEY'] = 'thisisasecretkey'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), nullable=False, unique=True)
    password = db.Column(db.String(80), nullable=False)

class RegisterForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    confirm = PasswordField(validators=[DataRequired(), EqualTo("password", message="Passwords must match."),],  render_kw={"placeholder": "Repeat password"})
    submit = SubmitField('Register')
    def validate_username(self, username):
        existing_user_username = User.query.filter_by(
            username=username.data).first()
        if existing_user_username:
            raise ValidationError(
                'That username already exists. Please choose a different one.')

class LoginForm(FlaskForm):
    username = StringField(validators=[InputRequired(), Length(min=4, max=20)], render_kw={"placeholder": "Username"})
    password = PasswordField(validators=[InputRequired(), Length(min=8, max=20)], render_kw={"placeholder": "Password"})
    submit = SubmitField('Login')

def validate(self):
        initial_validation = super(RegisterForm, self).validate()
        if not initial_validation:
            return False
        user = User.query.filter_by(username=self.username.data).first()
        if user:
            self.username.errors.append("Username already registered")
            return False
        if self.password.data != self.confirm.data:
            self.password.errors.append("Passwords must match")
            return False
        return True

def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))
    
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    faces = []
    locs = []
    preds = []
 
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
 
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))
 
    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)

def captureOnSmile():
    camera=cv2.VideoCapture(0)
    while(True):
        success,frame=camera.read()
        if not success:
            break
        else:
            faceCascade = cv2.CascadeClassifier(r"dataset\haarcascade_frontalface_default.xml")
            smileCascade = cv2.CascadeClassifier(r"dataset\haarcascade_smile.xml")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.3,5)
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    smiles=smileCascade.detectMultiScale(roi_gray,1.5,30,minSize=(50,50))
                    for i in smiles:
                        if len(i)>1:
                            cv2.putText(frame,"SMILING",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)
                            path=r"output\smile.jpg"
                            cv2.imwrite(path,frame)
                            camera.release()
                            cv2.destroyAllWindows()
                            break

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

def captureOnBlink():
    first_read = True
    b=0
    camera=cv2.VideoCapture(0)
    while(True):
        success,frame=camera.read()
        if not success:
            break
        else:
            faceCascade = cv2.CascadeClassifier(r"dataset\haarcascade_frontalface_default.xml")
            eye_cascade=cv2.CascadeClassifier(r"dataset\haarcascade_smile.xml")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.3,5)
            a=0
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    eyes = eye_cascade.detectMultiScale(roi_gray,1.3,5,minSize=(50,50)) 
                    if(len(eyes)>=1):
                        if(first_read):
                            a=1
                            cv2.putText(frame,"Eye detected ",(70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
                        else:
                            if(len(eyes)>=1):
                                if b==1:
                                    path=r'output\blink.jpg'
                                    cv2.imwrite(path,frame)
                                    camera.release()
                                    cv2.destroyAllWindows()
                                b=0
                                cv2.putText(frame,"blink ", (70,70),cv2.FONT_HERSHEY_PLAIN, 2,(123,123,255),2)
                    else:
                        if(first_read):
                            cv2.putText(frame,"ok", (70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                        else:
                            b=1
                            cv2.putText(frame,"okkkk", (70,70),cv2.FONT_HERSHEY_PLAIN, 3,(0,0,255),2)
                            first_read=True
                            break
            else:
                cv2.putText(frame,"No face detected",(100,100),cv2.FONT_HERSHEY_PLAIN, 3,(0,255,0),2)
                
            if(a==1 and first_read):
                first_read = False
            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

def captureOnGesture():
    mpHands = mp.solutions.hands
    hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mpDraw = mp.solutions.drawing_utils
    model = load_model('mp_hand_gesture')
    f = open('gesture.names', 'r')
    classNames = f.read().split('\n')
    f.close()
    print(classNames)

    camera = cv2.VideoCapture(0)

    while True:
        success,frame=camera.read()

        if not success:
            break
        else:
            x, y, c = frame.shape

            frame = cv2.flip(frame, 1)
            framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(framergb)
            
            className = ''

            if result.multi_hand_landmarks:
                landmarks = []
                for handslms in result.multi_hand_landmarks:
                    for lm in handslms.landmark:
                        lmx = int(lm.x * x)
                        lmy = int(lm.y * y)

                        landmarks.append([lmx, lmy])

                    prediction = model.predict([landmarks])
                    classID = np.argmax(prediction)
                    className = classNames[classID]

                    if className == "peace":
                        path=r"output\gesture.jpg"
                        cv2.imwrite(path,frame)
                        camera.release()
                        cv2.destroyAllWindows()
                        break

            cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0,0,255), 2, cv2.LINE_AA)

            if cv2.waitKey(1) == ord('q'):
                break

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')

def captureOnMask():
    prototxtPath = r"face_detector\deploy.prototxt"
    weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    maskNet = load_model("mask_detector.model")
 
    print("[INFO] starting video stream...")
    camera = cv2.VideoCapture(0)
    while True:
        success,frame=camera.read()
        
        if not success:
            break
        else:
            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            print(preds)
            for (box, pred) in zip(locs, preds):
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred
        
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        
                cv2.putText(frame, label, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                if mask > 0.90:
                    path=r"output\mask.jpg"
                    cv2.imwrite(path,frame)
                    camera.release()
                    cv2.destroyAllWindows()
                    break

            key = cv2.waitKey(1) & 0xFF
        
            if key == ord("q"):
                break

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def menu():
    if current_user.is_authenticated:
        return render_template('index.html')
    else:
        return redirect(url_for("login"))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        flash("You are already logged in.", "info")
        return redirect(url_for("menu"))
    form = LoginForm(request.form)
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if bcrypt.check_password_hash(user.password, form.password.data):
                login_user(user)
                return redirect(url_for('menu'))
            else:
                flash("Invalid username and/or password.", "danger")
                return render_template("login.html", form=form)
    return render_template('login.html', form=form)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        flash("You are already registered.", "info")
        return redirect(url_for("menu"))
    form = RegisterForm(request.form)
    if form.validate_on_submit():
        hashed_password = bcrypt.generate_password_hash(form.password.data)
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash("User Created", "success")
        return redirect(url_for('login'))

    return render_template('register.html', form=form)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    for filename in os.listdir("output"):
        filepath = os.path.join("output", filename)
        try:
            if os.path.isfile(filepath):
                os.remove(filepath)
        except Exception as e:
            print(f"Error deleting file {filepath}: {e}")
    flash("You were logged out.", "success")
    return redirect(url_for("login"))


@app.route('/selfie-on-smile.html')
def smile():
    path = r"output\smile.jpg"
    image_exists = os.path.exists(path) 
    return render_template('smile.html', image_exists=image_exists)

@app.route("/selfie-on-smile/download")
def smileDownload():
    path = r"output\smile.jpg"
    if os.path.isfile(path):
        return send_file(path, as_attachment=True)

@app.route('/selfie-on-blink.html')
def blink():
    path = r"output\blink.jpg"
    image_exists = os.path.exists(path) 
    return render_template('blink.html', image_exists=image_exists)

@app.route("/selfie-on-blink/download")
def blinkDownload():
    path = r"output\blink.jpg"
    if os.path.isfile(path):
        return send_file(path, as_attachment=True)

@app.route('/selfie-on-gesture.html')
def gesture():
    path = r"output\gesture.jpg"
    image_exists = os.path.exists(path) 
    return render_template('gesture.html', image_exists=image_exists)

@app.route("/selfie-on-gesture/download")
def gestureDownload():
    path = r"output\gesture.jpg"
    if os.path.isfile(path):
        return send_file(path, as_attachment=True)

@app.route('/selfie-on-mask.html')
def mask():
    path = r"output\mask.jpg"
    image_exists = os.path.exists(path) 
    return render_template('mask.html', image_exists=image_exists)

@app.route("/selfie-on-mask/download")
def maskDownload():
    path = r"output\mask.jpg"
    if os.path.isfile(path):
        return send_file(path, as_attachment=True)

@app.route('/VideoForSmile')
def VideoForSmile():
    return Response(captureOnSmile(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/VideoForBlink')
def VideoForBlink():
    return Response(captureOnBlink(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/VideoForGesture')
def VideoForGesture():
    return Response(captureOnGesture(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/VideoForMask')
def VideoForMask():
    return Response(captureOnMask(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
