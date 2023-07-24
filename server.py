import openai
import requests
from flask import jsonify, Flask, request
from flask_cors import CORS
import os
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
bing_search_api_key = os.getenv("BING_SEARCH_API")  # Add your Bing Search API key

MAX_CONVERSATION_MEMORY = 4
conversation_memory = []


def search(query):
    mkt = 'en-AU'
    params = {'q': query, 'mkt': mkt}  # Limit the number of search results to 1
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    try:
        response = requests.get(
            bing_search_endpoint,
            headers=headers,
            params=params
        )
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

        conversation_memory.append({"role": "Human", "content": word})
        conversation_memory.append({"role": "AI", "content": prompt})

        if len(conversation_memory) > 2 * MAX_CONVERSATION_MEMORY:
            conversation_memory = conversation_memory[-MAX_CONVERSATION_MEMORY:]

        if results:
            openai.api_key = openai_api_key

            try:
                formatted_prompt = format_prompt(conversation_memory)
                prompt_token_count = count_tokens(formatted_prompt)

                if prompt_token_count > 4096:
                    raise Exception("Token count exceeded the model's limit. Please ask a shorter question.")

                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=formatted_prompt,
                    max_tokens=3500, 
                    temperature=1.0,
                    n=1,
                    stop=None
                )

                response = response["choices"][0]["text"]
                conversation_memory.append({"role": "AI", "content": response})

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
