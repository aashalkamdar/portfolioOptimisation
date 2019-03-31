from flask import Flask,request

app = Flask(__name__)
app.config["DEBUG"]=True

@app.route('/home',methods=["GET","POST"])
def home():
    return str(request.form['stock'])

if __name__ == '__main__':
    app.run()