import openai
import requests
from flask import jsonify, Flask, request
from flask_cors import CORS
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

# Load the environment variables from the .env file
load_dotenv()

# Get the OpenAI API key from the environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
bing_search_api_key = os.getenv("BING_SEARCH_API")  # Add your Bing Search API key

# Set the maximum number of conversations to keep in memory
MAX_CONVERSATION_MEMORY = 4
# Initialize conversation memory as a list
conversation_memory = []


def search(query):
    mkt = 'en-AU'
    params = {'q': query, 'mkt': mkt}  # Limit the number of search results to 1
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    try:
        response = requests.get('https://api.bing.microsoft.com/bing/v7.0/search', headers=headers, params=params)
        response.raise_for_status()
        json = response.json()
        if json["webPages"]["value"]:
            snippet = json["webPages"]["value"][0]["snippet"][:400]  
            return snippet
        else:
            return None
    except requests.exceptions.RequestException as ex:
        raise Exception(f"Bing Search API Error: {ex}")


def format_prompt(messages):
    formatted_prompt = ""

    for message in messages:
        if message['role'] == 'Human':
            formatted_prompt += f"User: {message['content']}\n"
        elif message['role'] == 'AI':
            formatted_prompt += f"AI: {message['content']}\n"

    return formatted_prompt


def count_tokens(text):
    # Roughly estimate the number of tokens by splitting on whitespace
    tokens = text.split()
    return len(tokens)


@app.route('/api/home', methods=['GET', 'POST'])
def return_home():
    global conversation_memory

    if request.method == 'POST':
        data = request.get_json()
        word = data.get('word', '')

        results = search(word)

        prompt = "Use these sources to answer the question:\n\n" + \
                 "Source:\n" + results + "\n\nQuestion: " + word + "\n\nAnswer:"

        # Save the user input and AI response in the conversation memory
        conversation_memory.append({"role": "Human", "content": word})
        conversation_memory.append({"role": "AI", "content": prompt})

        # Remove older conversations if the conversation memory exceeds the maximum limit
        if len(conversation_memory) > 2 * MAX_CONVERSATION_MEMORY:
            conversation_memory = conversation_memory[-MAX_CONVERSATION_MEMORY:]

        if results:
            openai.api_key = openai_api_key

            try:
                # Generate the prompt using the conversation history
                formatted_prompt = format_prompt(conversation_memory)
                prompt_token_count = count_tokens(formatted_prompt)

                if prompt_token_count > 4096:
                    raise Exception("Token count exceeded the model's limit. Please ask a shorter question.")

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=formatted_prompt,
                    max_tokens=3500,  # Limit the maximum tokens to prevent exceeding the model's limit
                    temperature=1.0,
                    n=1,
                    stop=None
                )

                response = response["choices"][0]["text"]
                # Save the AI response in the conversation memory
                conversation_memory.append({"role": "AI", "content": response})

                # Extract the AI response content without the "AI:" prefix
                response_content = response.split(": ", 1)[-1].strip()

                return jsonify({
                    'message': response_content
                })
            except Exception as ex:
                return jsonify({
                    'message': f"An error occurred while processing your request: {ex}"
                })

        else:
            return jsonify({
                'message': "No results found for the given query."
            })
    else:
        return jsonify({
            'message': 'Henry Moragan'
        })


if __name__ == "__main__":
    app.run(debug=True)
