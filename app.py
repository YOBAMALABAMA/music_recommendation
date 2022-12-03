from flask import Flask, render_template, request 


app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/sub", methods=["POST"])
def sub():
    if request.method == "POST":
        name = request.form['artist'] 
    
    return render_template('sub.html', n = name)     


if __name__=="__main__":
    app.run(debug=True)    