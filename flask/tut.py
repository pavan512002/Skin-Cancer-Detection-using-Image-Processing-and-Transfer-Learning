from flask import Flask,render_template,request,send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

class_dict ={0:"Basal_Cell_Carcinoma (Cancer)",
             1:"Melanoma (Cancer)",
             2:"Nevus (Non-Cancerous)"}
model = load_model('C:/Users/pavan/Downloads/ISIC_2019_Training_Input/Nevus/model_v1.h5')

def pred_image(img_path, model, class_dict):
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32')/255

    preds = model.predict(img_array)[0]
    class_probs = []
    max_prob = 0
    max_class = ''
    for i in range(len(class_dict)):
        class_name = class_dict[i]
        prob = round(preds[i]*100, 2)
        class_probs.append((class_name, prob))
        if prob > max_prob:
            max_prob = prob
            max_class = class_name

    print(f"Prediction: {max_class}")
    return f"Prediction: {max_class}"

app = Flask(__name__)

@app.route("/",methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file found"
        file = request.files['file']
        if file.filename == '':
            return "No file selected"
        if file:
            file_path = "C:/Users/pavan/PycharmProjects/pythonProject/flask/static/uploads/" + file.filename
            file.save(file_path)
            result = pred_image(file_path,model,class_dict)
            # res = prediction + " detected " + precaution
            return render_template("result.html", result=result,image_file = file.filename)
    return render_template("index.html")
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('static/uploads',filename)



app.run(debug = True)