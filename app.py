import streamlit as st

import json
from haystack import Document, Pipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever, PromptTemplate, PromptNode
from pprint import pprint
from io import StringIO
import time


HF_TOKEN = "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"
    
@st.cache_resource
def create_description(item_name, details):
    description = f"{item_name}: "
    if isinstance(details, list):
        description += f"{details[0]}, Price: {details[1]}, "
        if len(details) > 2 and isinstance(details[2], dict):
            for key, value in details[2].get("nutritionalInfo", {}).items():
                description += f"{key}: {value}, "
            description += f"Available: {details[2]['available']}"
    elif isinstance(details, dict):
        description += f"Name: {details.get('name', '')}, Price: {details.get('price', '')}"
        if 'contents' in details:
            description += ", Contents: ["
            for item in details['contents']:
                if isinstance(item, list):
                    description += f"({item[0]}, {item[1]}), "
                elif isinstance(item, dict):
                    description += f"({item.get('from', '')}, Size: {item.get('size', '')}), "
            description = description.rstrip(", ")
            description += "]"
    return description

    
@st.cache_resource
def injection(menu):
    with open(menu, 'r') as file:
        menu_data = json.load(file)
    documents = []
    
    for category, items in menu_data.items():
        for item_name, details in items.items():
            description = create_description(item_name, details)
            documents.append(Document(content=description))
    
    document_store = InMemoryDocumentStore()
    document_store.write_documents(documents)
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",)
    document_store.update_embeddings(retriever)
    return retriever

def addData(file):

    stringio = StringIO(file.getvalue().decode("utf-8"))
    file_content = stringio.read()
    with open('data/uploaded.json', 'w') as filez:
        filez.write(file_content)
    
    document_store = injection('menu.json').document_store

    with open('data/uploaded.json', 'r') as filez:
        data = json.load(filez)
        
    documents = []
    for category, items in data.items():
        for item_name, details in items.items():
            description = create_description(item_name, details)
            documents.append(Document(content=description))
        
    document_store.write_documents(documents)
    
    retriever = EmbeddingRetriever(document_store=document_store, embedding_model="deepset/sentence_bert",)
    
    document_store.update_embeddings(retriever)
    
@st.cache_resource
def pipeline_initialization():
    pn = PromptNode(model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
                max_length=800,
                api_key=HF_TOKEN)
    
    retriever = injection('menu.json')
    qa_template = PromptTemplate(prompt=
      """<s>[INST] You are working as a drive-in cashier at a fast-food restaurant act like such, be accommodating and be careful about what you are asked without writing too much. Using the information contained in the context, answer the question. If the question is not fast-food related remember the customer your function.
      If the answer cannot be deduced from the context, answer \"I don't know.\"
      Context: {join(documents)};
      Question: {query}
      [/INST]""")
    prompt_node = PromptNode(model_name_or_path="mistralai/Mixtral-8x7B-Instruct-v0.1",
                         api_key=HF_TOKEN,
                         default_prompt_template=qa_template,
                         max_length=5500,
                         model_kwargs={"model_max_length":8000})
    rag_pipeline = Pipeline()
    rag_pipeline.add_node(component=retriever, name="retriever", inputs=["Query"])
    rag_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])
        
    return rag_pipeline

print_answer = lambda out: pprint(out["results"][0].strip())

@st.cache_resource
def timeData():
    return []

st.title("KFC ordering assistant! :poultry_leg:")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.sidebar:
        st.subheader("🗒️ Your documents")
        new_docs = st.file_uploader("Upload your json here and click on 'Process'", accept_multiple_files=False)
        if st.button("Process"):
            with st.spinner("Processing"):
                addData(new_docs)
        st.subheader("⚙️ Options")
                
        def reset_conversation():
            st.session_state.conversation = None
            st.session_state.chat_history = None
            st.session_state.messages.clear()
            
        st.button('Reset Chat', on_click=reset_conversation)

if prompt := st.chat_input("How may I be of help today?"):

    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        rag_pipeline=pipeline_initialization()
        response = []
        start_time = time.time()
        result = rag_pipeline.run(query=prompt)['results'][0]
        end_time = time.time()
        timeData().append((end_time - start_time)*1000)
        response.append(st.write(result))
        with st.expander('Statistics'):
            final_time = (end_time - start_time)*1000
            st.write(f'This execution took: %.2f ms and it was request number: {len(timeData())}.' % final_time)
            st.line_chart(data=timeData())
            
    st.session_state.messages.append({"role": "assistant", "content": result})
