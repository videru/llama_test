from flask import Flask, request, jsonify
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/send-message', methods=['POST'])
def send_message():
    try:
        user_input = request.json['message']
        
        # Ensure that the user input is not empty
        if not user_input.strip():
            return jsonify({'reply': 'Please enter a message.'})

        # Call GPT model here and get the response
        response = "hi"
        # Extract and send back the response
        return jsonify({'reply': response.strip()})

    except Exception as e:
        # Log the exception to the console or a file for debugging
        print(f"An error occurred: {e}")
        return jsonify({'reply': 'An error occurred while processing your request.'})

if __name__ == '__main__':
    app.run(debug=True)
