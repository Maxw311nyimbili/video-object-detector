from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import StringField, SubmitField
from wtforms.validators import DataRequired

# pip install WTForms
# pip install Flask-WTF


class UploadForm(FlaskForm):
    video = FileField('Upload Video', validators=[
        FileRequired(),
        FileAllowed(['mp4', 'avi', 'mov'], 'Videos only!')
    ])
    search_query = StringField('Search Query', validators=[DataRequired()])
    submit = SubmitField('Submit')
