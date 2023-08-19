from flask import Flask, request, render_template
import numpy as np
import keras

app = Flask('__name__')

@app.route('/')
def read_main():
    return render_template('index.html')

@app.route('/predict', methods=['GET'])
def generate_output():
    json_data = False
    input_data = request.args.get('data')
    if input_data is None:
        input_data = request.get_json()
        json_data = True
    predicted_price = process_and_predict(input_data=input_data, json_data=json_data)
    return {'predicted_price': predicted_price}

def process_and_predict(input_data, json_data):
    if json_data:
        output_data = [float(item) for item in input_data['data'].split(',')]
    else:
        output_data = [float(item) for item in input_data.split(',')]

    output_data[3],output_data[8] = int(output_data[3]),int(output_data[8])
    # Transform input data
    output_data = np.array(output_data).reshape(1, -1)
    output_data_transformed = keras.utils.normalize(output_data)

    # Load the model
    model = keras.models.load_model('src/models/model.h5')

    # Make predictions
    predicted_price = model.predict(output_data_transformed).flatten()

    return float(predicted_price[0])  # Convert to float for JSON response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
