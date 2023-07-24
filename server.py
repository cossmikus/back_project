# from flask import Flask, jsonify
# from flask_cors import CORS

# # app instance
# app = Flask(__name__)
# CORS(app)

# @app.route('/api/home', methods=['GET'])
# def return_home():
#     return jsonify({
#         'message': 'Henry Moragan'
#     })


# if __name__ == '__main__':
#     app.run(debug=True, port = 8080)




# from flask import Flask, jsonify, request
# from flask_cors import CORS

# # app instance
# app = Flask(__name__)
# CORS(app)

# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')
#         return jsonify({
#             'message': word+'Gave'
#         })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)







# import os
# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question, act like a financial asssistant, that answers only financial questions. reply hello to hello. reject wuestions other than finance:\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         if results:
#             openai.api_key = openai_api_key

#             response = openai.Completion.create(
#                 engine="text-davinci-003",
#                 prompt=prompt,
#                 max_tokens=500,  # Set the desired maximum tokens
#                 temperature=1.0,
#                 n=1,
#                 stop=None
#             )

#             response = response["choices"][0]["text"]
#             return jsonify({
#                 'message': response
#             })
#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)


# import os
# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.vectorstores.base import Document

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be '  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# # Initialize LangChain components
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Load data into LangChain Document format
# data = [
#     Document(page_content="Your first text"),
#     Document(page_content="Your second text"),
#     Document(page_content="Your third text")
# ]

# # Create the FAISS vectorstore from the documents and embeddings
# vectors = FAISS.from_documents(data, embeddings)

# # Create the ConversationalRetrievalChain with ChatOpenAI model and the vectorstore as the retriever
# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
#     retriever=vectors.as_retriever()
# )


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question, act like a financial assistant, that answers only financial questions. Reply hello to hello. Reject questions other than finance:\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         if results:
#             response = chain({"question": word, "chat_history": [], "prompt": prompt})

#             return jsonify({
#                 'message': response["answer"]
#             })
#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)


# import os
# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.vectorstores.base import Document

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# # Initialize LangChain components
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Load data into LangChain Document format
# data = [
#     Document(page_content="Your first text"),
#     Document(page_content="Your second text"),
#     Document(page_content="Your third text")
# ]

# # Create the FAISS vectorstore from the documents and embeddings
# vectors = FAISS.from_documents(data, embeddings)

# # Create the ConversationalRetrievalChain with ChatOpenAI model and the vectorstore as the retriever
# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
#     retriever=vectors.as_retriever()
# )


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question, Reply hello to hello:\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         try:
#             if results:
#                 response = chain({"question": word, "chat_history": [], "prompt": prompt})
#                 return jsonify({
#                     'message': response["answer"]
#                 })
#             else:
#                 return jsonify({
#                     'message': "No results found for the given query."
#                 })
#         except openai.error.APIError as e:
#             if "exceeds the max tokens limit" in str(e):
#                 return jsonify({
#                     'message': "The question is too long. Please ask a more concise question."
#                 })
#             else:
#                 return jsonify({
#                     'message': "An error occurred while processing the question. Please try again later."
#                 })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })



# import os
# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.chat_models import ChatOpenAI
# from langchain.chains import ConversationalRetrievalChain
# from langchain.vectorstores import FAISS
# from langchain.vectorstores.base import Document

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Access OpenAI API key from the environment
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Access Bing Search API key from the environment
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# # Initialize LangChain components
# embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# # Load data into LangChain Document format
# data = [
#     Document(page_content="Your first text"),
#     Document(page_content="Your second text"),
#     Document(page_content="Your third text")
# ]

# # Create the FAISS vectorstore from the documents and embeddings
# vectors = FAISS.from_documents(data, embeddings)

# # Create the ConversationalRetrievalChain with ChatOpenAI model and the vectorstore as the retriever
# chain = ConversationalRetrievalChain.from_llm(
#     llm=ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo', openai_api_key=openai_api_key),
#     retriever=vectors.as_retriever()
# )


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# def truncate_prompt(prompt, max_tokens):
#     prompt = prompt.strip()
#     prompt_tokens = prompt.split(" ")
#     truncated_tokens = []
#     token_count = 0

#     for token in prompt_tokens:
#         token_count += len(token.split())
#         if token_count <= max_tokens:
#             truncated_tokens.append(token)
#         else:
#             break

#     return " ".join(truncated_tokens)


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         if results:
#             results_prompts = [
#                 f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#             ]
#             prompt = "Use these sources to answer the question, act like a financial assistant, that answers only financial questions. Reply hello to hello. Reject questions other than finance:\n\n" + \
#                      "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#             # Truncate the prompt if the total tokens exceed 4096 for GPT-3.5 API
#             max_tokens = 4096
#             prompt = truncate_prompt(prompt, max_tokens)
#         else:
#             prompt = "Act like a financial assistant, that answers only financial questions. Reply hello to hello. Reject questions other than finance:\n\n" + \
#                      "Question: " + word + "\n\nAnswer:"

#         try:
#             if results:
#                 response = chain({"question": word, "chat_history": [], "prompt": prompt})
#                 return jsonify({
#                     'message': response["answer"]
#                 })
#             else:
#                 return jsonify({
#                     'message': "No results found for the given query."
#                 })
#         except openai.error.APIError as e:
#             if "exceeds the max tokens limit" in str(e):
#                 return jsonify({
#                     'message': "The question is too long. Please ask a more concise question."
#                 })
#             else:
#                 return jsonify({
#                     'message': "An error occurred while processing the question. Please try again later."
#                 })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)





# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question, act like a financial asssistant, that answers only financial questions. reply hello to hello. reject wuestions other than finance:\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         if results:
#             openai.api_key = openai_api_key

#             response = openai.Completion.create(
#                 engine="text-davinci-003",
#                 prompt=prompt,
#                 max_tokens=500,  # Set the desired maximum tokens
#                 temperature=1.0,
#                 n=1,
#                 stop=None
#             )

#             response = response["choices"][0]["text"]
#             return jsonify({
#                 'message': response
#             })
#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)



# import os
# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question. reply hello to hello :\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         if results:
#             openai.api_key = openai_api_key

#             response = openai.Completion.create(
#                 engine="text-davinci-003",
#                 prompt=prompt,
#                 max_tokens=3000,  # Set the desired maximum tokens
#                 temperature=1.0,
#                 n=1,
#                 stop=None
#             )

#             response = response["choices"][0]["text"]
#             return jsonify({
#                 'message': response
#             })
#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)


# import openai
# import requests
# from pprint import pprint
# import textwrap
# from flask import jsonify, Flask, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question. reply hello to hello :\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         if results:
#             openai.api_key = openai_api_key

#             try:
#                 response = openai.Completion.create(
#                     engine="text-davinci-003",
#                     prompt=prompt,
#                     max_tokens=1500,  # Set a value that should exceed the limit
#                     temperature=1.0,
#                     n=1,
#                     stop=None
#                 )

#                 response = response["choices"][0]["text"]
#                 return jsonify({
#                     'message': response
#                 })
#             except openai.error.InvalidRequestError as e:
#                 # The token count exceeded the model's limit, inform the frontend to ask again
#                 return jsonify({
#                     'message': "Token count exceeded the model's limit. Please ask a shorter question."
#                 })
#             except Exception as ex:
#                 return jsonify({
#                     'message': "An error occurred while processing your request."
#                 })

#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)



# import openai
# import requests
# from flask import jsonify, Flask, request
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# openai_api_key = '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'  # Add your OpenAI API key
# bing_search_api_key = '387dd4f546fc49eea02bc4d57df1f5be'  # Add your Bing Search API key
# bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# # Initialize conversation memory as a list
# conversation_memory = []


# def search(query):
#     mkt = 'en-US'
#     params = {'q': query, 'mkt': mkt}
#     headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

#     try:
#         response = requests.get(bing_search_endpoint, headers=headers, params=params)
#         response.raise_for_status()
#         json = response.json()
#         return json["webPages"]["value"]
#     except Exception as ex:
#         raise ex


# def format_prompt(messages):
#     formatted_prompt = "The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know."

#     for message in messages:
#         if message['role'] == 'Human':
#             formatted_prompt += f"\nHuman: {message['content']}"
#         elif message['role'] == 'AI':
#             formatted_prompt += f"\nAI: {message['content']}"

#     return formatted_prompt


# @app.route('/api/home', methods=['GET', 'POST'])
# def return_home():
#     if request.method == 'POST':
#         data = request.get_json()
#         word = data.get('word', '')

#         results = search(word)

#         results_prompts = [
#             f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}" for result in results
#         ]

#         prompt = "Use these sources to answer the question. reply hello to hello, hi to hi :\n\n" + \
#                  "\n\n".join(results_prompts) + "\n\nQuestion: " + word + "\n\nAnswer:"

#         # Save the user input and AI response in the conversation memory
#         conversation_memory.append({"role": "Human", "content": word})
#         conversation_memory.append({"role": "AI", "content": prompt})

#         if results:
#             openai.api_key = openai_api_key

#             try:
#                 # Generate the prompt using the conversation history
#                 formatted_prompt = format_prompt(conversation_memory)

#                 response = openai.Completion.create(
#                     engine="text-davinci-003",
#                     prompt=formatted_prompt,
#                     max_tokens=1500,  # Set a value that should exceed the limit
#                     temperature=1.0,
#                     n=1,
#                     stop=None
#                 )

#                 response = response["choices"][0]["text"]

#                 # Save the AI response in the conversation memory
#                 conversation_memory.append({"role": "AI", "content": response})

#                 return jsonify({
#                     'message': response
#                 })
#             except openai.error.InvalidRequestError as e:
#                 # The token count exceeded the model's limit, inform the frontend to ask again
#                 return jsonify({
#                     'message': "Token count exceeded the model's limit. Please ask a shorter question."
#                 })
#             except Exception as ex:
#                 return jsonify({
#                     'message': "An error occurred while processing your request."
#                 })

#         else:
#             return jsonify({
#                 'message': "No results found for the given query."
#             })
#     else:
#         return jsonify({
#             'message': 'Henry Moragan'
#         })


# if __name__ == '__main__':
#     app.run(debug=True, port=8080)

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
bing_search_endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# Set the maximum number of conversations to keep in memory
MAX_CONVERSATION_MEMORY = 4
# Initialize conversation memory as a list
conversation_memory = []


def search(query):
    mkt = 'en-US'
    params = {'q': query, 'mkt': mkt, 'count': 1}  # Limit the number of search results to 1
    headers = {'Ocp-Apim-Subscription-Key': bing_search_api_key}

    try:
        response = requests.get(bing_search_endpoint, headers=headers, params=params)
        response.raise_for_status()
        json = response.json()
        if json["webPages"]["value"]:
            # Extract only the important information (snippet) from the search result and truncate it
            snippet = json["webPages"]["value"][0]["snippet"][:400]  # Truncate to reduce tokens
            return snippet
        else:
            return None
    except Exception as ex:
        raise ex


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
            except openai.error.InvalidRequestError as e:
                # The token count exceeded the model's limit, inform the frontend to ask again
                clear_conversation_memory()  # Clear conversation history
                return jsonify({
                    'message': "Token count exceeded the model's limit. Please ask a shorter question."
                })
            except Exception as ex:
                return jsonify({
                    'message': "An error occurred while processing your request."
                })

        else:
            return jsonify({
                'message': "No results found for the given query."
            })
    else:
        return jsonify({
            'message': 'Henry Moragan'
        })


def clear_conversation_memory():
    global conversation_memory
    conversation_memory = []


if __name__ == '__main__':
    app.run(debug=True, port=8080)
