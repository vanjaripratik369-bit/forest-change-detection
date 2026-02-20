from flask import Flask, render_template, request
import os
from detector import analyze_forest_change, analyze_single_image_ml



app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":

        before = request.files.get("before")
        after = request.files.get("after")

        # 🟢 CASE 1: BOTH images uploaded
        if before and after:
            before_path = os.path.join(UPLOAD_FOLDER, before.filename)
            after_path = os.path.join(UPLOAD_FOLDER, after.filename)

            before.save(before_path)
            after.save(after_path)

            result = analyze_forest_change(before_path, after_path)

            return render_template(
                "result.html",
                result=result,
                before_img=before.filename,
                after_img=after.filename
            )

        # 🔵 CASE 2: ONLY ONE image uploaded
        elif before or after:
            image = before if before else after
            image_path = os.path.join(UPLOAD_FOLDER, image.filename)
            image.save(image_path)

            result = analyze_single_image_ml(image_path)


            return render_template(
                "single_result.html",
                result=result,
                image=image.filename
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
