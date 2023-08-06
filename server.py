
import requests
import openai
from flask import jsonify, Flask, request
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

openai.api_key = os.getenv("OPENAI_API_KEY")  # Add your OpenAI API key

# Create a variable to store the user's previous input
previous_user_input = ""


@app.route('/api/home', methods=['POST'])
def return_home():
    global previous_user_input
    data = request.get_json()
    user_input = data.get('word', '')
    
    messages = [
        {
            "role": "system",
            "content": "you are a helpful assistant who recommends movies and TV shows"
        },
        {
            "role": "system",
            "content": "ask user what kind of movies or TV shows they would like to watch"
        },
        {
            "role": "system",
            "content": "include short description, release year and IMDb rating of a movie"
        },
        {
            "role": "system",
            "content": "write recommendations step-by-step. include spaces, next line and etc."
        },
        {
            "role": "system",
            "content": "Recommend minimum 10 movies or TV shows, ask the user if he/she wants more movie recommendation in this genre"
        },
        {
            "role": "system",
            "content": "Ask the user about their favorite movie or TV show genre, actor, or director to personalize the recommendations. Inquire if the user has any specific mood or theme in mind for the movie or TV show they want to watch."
        },
        {
            "role": "user",
            "content": user_input
        },       
    ]

    try:

        response = openai.ChatCompletion.create(
            model="gpt-4-0613",
            messages=messages,
            temperature=1,
            max_tokens=3300,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        response_text = response['choices'][0]['message']['content']
        
        # Update the previous_user_input with the latest user input
        previous_user_input = user_input
        
        return jsonify({
            'message': response_text
        })

    except openai.error.InvalidRequestError as e:
        return jsonify({
            'message': f"Invalid request to OpenAI: {e}"
        })

    except Exception as ex:
        return jsonify({
            'message': f"An error occurred: {ex}"
        })


@app.route('/api/clear', methods=['POST'])
def clear_conversation():
    global previous_user_input
    # Clear the previous_user_input when the conversation is cleared in the frontend
    previous_user_input = ""
    return jsonify({'message': 'Conversation cleared successfully.'})


