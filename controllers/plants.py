import os
from werkzeug.utils import secure_filename
from forms.plant_disease_form import UploadForm
from models.plant import PlantDiseaseModel
from flask import Blueprint, render_template, request, current_app

def lession_cnn():
    model = PlantDiseaseModel()
    form = UploadForm()
    prediction = None
    confidence = None
    user_image = None

    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction, confidence = model.predict(filepath)
        user_image = filepath

    return render_template('lession_cnn.html', form=form, prediction=prediction, confidence=confidence, user_image=user_image)
