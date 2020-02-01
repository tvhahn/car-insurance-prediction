from flask import Flask
from flask import request

 
app = Flask(__name__)
 
from flask import Flask
from flask import request
  
app = Flask(__name__)
  
@app.route('/query_example')
def query_example():
    request
    return 'Todo ...'

@app.route('/json_example',methods=['POST'])
def json_example():
    req_data = request.get_json()

    device = req_data['device']
    val_temp = req_data['value']
    timestamp = req_data['timestamp']

    return '''<h1>device {}<h1>'''.format(device)

# @app.route('/json_example',methods=['GET'])
# def json_view():

#     return "device {}".format(device)



if __name__ == '__main__':
    app.run(debug=True, port=5000)

# @app.route('/')
# def hello_world():
#     return "Hello world!!!!"

# @app.route("/postjson", methods = ['POST'])
# def postJsonHandler():
#     print (request.is_json)
#     content = request.get_json()
#     print (content)
#     return 'JSON posted'

# app.run(host='0.0.0.0', port= 8090)


# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8090)
