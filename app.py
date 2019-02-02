from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/')
def index():
  return render_template("home.html")


@app.route('/profile/<username>')
def profile(username, topten):
  return render_template("profile_topten.html", username = username, topten = topten )
  #topten is list of 10 book objects






if __name__ == "__main__":
    app.run(debug = True)

