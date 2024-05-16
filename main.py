from flask import Flask, render_template, request
import AuxFun, TCM, RM, SA, GC, Tree, threading, time

app = Flask(__name__)

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/sub', methods=['POST'])
def sub():
    result = request.form
    Texto = result.get('text')
    
    
    return render_template('sub.html'  )
