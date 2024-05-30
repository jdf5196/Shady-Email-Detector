import os
from flask import Flask, render_template

# TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), 'templates')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')