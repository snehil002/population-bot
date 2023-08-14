import os
import json
import openai
import gradio as gr
import time
import langchain
# from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
# from langchain.indexes import VectorstoreIndexCreator
# from langchain.retrievers.document_compressors import EmbeddingsFilter
# from langchain.retrievers import ContextualCompressionRetriever

openai.api_key = os.environ['OPENAI_API_KEY']

def get_completion(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    return response.choices[0].message["content"] #, response.usage

def read_string_to_list(input_string):
    if input_string is None:
        return None

    try:
        input_string = input_string.replace("'", "\"")  # Replace single quotes with double quotes for valid JSON
        data = json.loads(input_string)
        return data
    except json.JSONDecodeError:
        print("Error: Invalid JSON string")
        return None

file_name = "./population_mod.csv"

loader = CSVLoader(file_path=file_name)
docs = loader.load()

embeddings = OpenAIEmbeddings()

db = DocArrayInMemorySearch.from_documents(
    docs,
    embeddings
)

def create_chat_str(chat_hist_ui):
    chat_str = ""
    for h in chat_hist_ui:
        usermsg, assmsg = h
        chat_str += f"Human: {usermsg}\n"
        chat_str += f"Assistant: {assmsg}\n"
    return chat_str

# Create a standalone question
def create_standalone_question(user_input, chat_history):
    user_message_1 = f"""
You will be provided with a Chat history and a follow up question.
Rephrase the follow up question to be a stand alone question.

Chat history:
{chat_history}

Follow up question:
{user_input}

Standalone question:
"""
    messages = [
        {"role": "user", "content": user_message_1}
    ]
    answer = get_completion(messages)
    print("standalone: ", answer)
    return answer

def extract_country_and_columns(user_input):
    delimiter = "####"
    # user_input = query

    system_message = """
You will be given a user query delimited by ####. 
You need to extract a list of unique country names from the user query.
Convert the country names to the internationally used names.
Also, identify a list of columns that are required to answer the query.
Output a JSON with the key: countries, columns. Dont output any explainations.

Columns of the Population data CSV:
country (required)
population
growth_rate
world_percentage
area_sq_km
pop_density
rank
"""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{user_input}{delimiter}"}
    ]

    response = get_completion(messages)
    return response

def search_vectordb_by_country(query):
    res = db.max_marginal_relevance_search(query, k=1)
    return res[0].page_content

def get_relevant_info(res):
    countries = res["countries"]
    req_columns = res["columns"]
    info_bl = []
    for country in countries:
        rowstr = search_vectordb_by_country(country)
        print(rowstr)
        cdata = {f"{c.split(': ')[0]}" : c.split(": ")[1] for c in rowstr.split("\n")}
        if "country" not in req_columns:
            req_columns += ["country"]
        info_l = []
        for col in req_columns:
            info_l += [col + ": " + cdata[col]]
        info_bl += ["\n".join(info_l)]
    info = "\n\n".join(info_bl)
    return info

def get_final_answer(user_input, rel_info):
    delimiter = "####"
    # user_input = "is the population count of japan more than that of uk, pakistan or uae"#query
    # rel_info = info

    system_message = f"""Use the relevant information to answer the user query.
Answer in a polite and helpful tone with very concise answers.
Make sure to ask the user relevant follow up questions.
User query will be delimited by {delimiter} characters."""

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"{delimiter}{user_input}{delimiter}"},
        {"role": "system", "content": f"Relevant population data:\n{rel_info}"}
    ]

    response = get_completion(messages)
    # call_history += [response[1]]
    return response

def process_input(user_input, context):
    if user_input.strip() == "":
        raise Exception("No input")

#     tokens = int(tokens)
    if len(context) > 0:
        chat_history = create_chat_str(context)
        time.sleep(10)
        user_input = create_standalone_question(user_input, chat_history) # LLM
#         print(response_tup)
#         user_input = response_tup[0]
#         tokens += response_tup[1]["total_tokens"]
    
    time.sleep(10)
    response_str = extract_country_and_columns(user_input) # LLM
#     print(response_tup)
#     tokens += response_tup[1]["total_tokens"]
    response = read_string_to_list(response_str)
    rel_info = get_relevant_info(response)
#     print(rel_info)
    
    time.sleep(10)
    answer = get_final_answer(user_input, rel_info) # LLM
#     print(response_tup)
#     tokens += response_tup[1]["total_tokens"]
    return answer
#     except Exception as exp:
#         return exp.args

def ui_func(user, history):
    return "", history + [[user, None]]

def ui_func_2(history):
    user = history.pop()[0]
    try:
        response = process_input(user, history)
        history.append([user, response])
        return "", history
    except Exception as exp:
        print("Error!")
        print(exp.args)
        gr.Warning(str(exp.args))
        return user, history

with gr.Blocks() as demo:
    gr.Markdown("""# Chat with Population Data 2022
Ask any question in the input field. Press Enter to Send. 😇 History remains on this page!""")
    
    chatbot = gr.Chatbot(label="Chat History", height=400)
    msg = gr.Textbox(label="User Input", placeholder="Enter your question")
    clear = gr.ClearButton([msg, chatbot])
    
    msg.submit(ui_func, [msg, chatbot], [msg, chatbot], queue=False).then(
      ui_func_2, chatbot, [msg, chatbot]
    )

demo.queue()
demo.launch(show_error=True)
