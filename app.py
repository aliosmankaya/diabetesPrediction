from flask import Flask, render_template, redirect, request, url_for
from utils.predict import predict, columns, model


app = Flask(
    __name__,
    static_url_path='',
    static_folder='TEMPLATES',
    template_folder="TEMPLATES"
    )

app.secret_key = "\x84/!\x16&\xe6\xeb\x8an\n\na\re\xb3\x11\xe5\xf1\xf6\xdd\xf8\xed\x05G"


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/", methods=["GET","POST"])
def prediction():

    if request.method == "POST":

        checks = request.form.getlist("options")
        age = checks[0]
        pred = predict(columns, age, checks, model)       

        return render_template("index.html", pred=pred)

    else:
        return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)