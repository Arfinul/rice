from flask import Flask, request, Response
import jsonpickle
import os

# Initialize the Flask application
app = Flask(__name__)
# cwd = '/home/agnexttech/trap_next/images'
cwd = 'C:\\Users\Agnext-Arfin\Desktop\\uploaded'


# Route http posts to this method
@app.route('/trap_next/upload', methods=['POST'])
def test():
    os.chdir(cwd)
    print('Uploading the file ... Wait !!!')

    file = request.files['image']
    val = request.form['text']
    file.save(file.filename)
    # print(file.filename)
    print('Uploaded - ', file.filename)
    response = {'Image Uploaded', val
                }
    response_pickled = jsonpickle.encode(response)
    return Response(response=response_pickled, status=200, mimetype="application/json")


# start flask app
app.run(port=4002)
# app.run(host="0.0.0.0", port=5001)
