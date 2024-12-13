import streamlit as st
import time
import google.generativeai as genai
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import json
from tqdm import tqdm
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
import numpy as np
import subprocess
import getpass


# Load environment variables
load_dotenv()
dataset = "aws-service-logs"

def preprocess_corpus(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(input_file, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
            doc = {
                "id": f"{i}",  # Changed to match qrels format
                "contents": line.strip()
            }
            with open(os.path.join(output_dir, f"doc{i}.json"), 'w') as out:
                json.dump(doc, out)

def build_index(input_dir, index_dir):
    if os.path.exists(index_dir) and os.listdir(index_dir):
        #print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw"
    ]
    subprocess.run(cmd, check=True)

def load_queries(query_file):
    with open(query_file, 'r') as f:
        return [line.strip() for line in f]

def search(searcher, queries, top_k=10, query_id_start=0):
    results = {}
    searchResult = []
    for i, query in enumerate(tqdm(queries, desc="Searching")):
        hits = searcher.search(query, k=top_k)
        for hit in hits:
            if hit.score > 0.57:
                searchResult.append((hit.lucene_document.get('raw')))
                
    results[str(i + query_id_start)] = searchResult
    return results

def mainSearch(dataset,query):
    """main function for searching"""

    cname = dataset

    base_dir = f"data/logs/"
    query_id_start = {
        "aws-service-logs": 1,
    }[cname]

    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = os.path.join(base_dir, f"{cname}.dat")

    query_file = os.path.join(base_dir, f"{cname}-queries.txt")

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    #else:
    #    print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)

    # Search
    searcher = LuceneSearcher(index_dir)
    
    searcher.set_bm25(k1=15, b=0.4)


    #print('query is -', query )
    results = search(searcher, query, query_id_start=query_id_start)



    searchResult = list(results.items())[0]
    #print('print searchResult: ', searchResult)
    contextDocument = []
    for contentTuple in searchResult[1]:

        r2 = contentTuple.split(',')
        r3=r2[1]
        r4 = r3.split(' : ')
        contextDocument.append(r4[1])

    #print(contextDocument)
    #print('query[0]', query[0])
    #print('queryfile:',  query_file)

    # Save results
    with open(query_file, "a") as f:
        f.write(f'\n{query[0]}')

  

    # load Cache
    cacheResult = []
    with open("cacheResults_aws-service-logs.json", "r") as f:
        cacheResult=json.load(f)

    cacheResult[str(len(cacheResult))] =  contextDocument

    # Update Cache
    with open("cacheResults_aws-service-logs.json", "w") as f:
        json.dump(cacheResult, f)
    
    return contextDocument

def loadCache(dataset):
    """ Loading cache"""

    cname = dataset

    base_dir = f"data/logs/"
    query_id_start = {
        "aws-service-logs": 1,
    }[cname]

    # Paths to the raw corpus, queries, and relevance label files
    corpus_file = os.path.join(base_dir, f"{cname}.dat")
    query_file = os.path.join(base_dir, f"{cname}-queries.txt")
    # processed_corpus_dir = os.path.join(base_dir, "corpus")

    # Directories where the processed corpus and index will be stored for toolkit
    processed_corpus_dir = f"processed_corpus/{cname}"
    os.makedirs(processed_corpus_dir, exist_ok=True)
    index_dir = f"indexes/{cname}"

    # Preprocess corpus
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}. Skipping preprocessing.")

    # Build index
    build_index(processed_corpus_dir, index_dir)

    # Load queries and qrels
    queries = load_queries(query_file)

    # Search
    searcher = LuceneSearcher(index_dir)
    
    searcher.set_bm25(k1=15, b=0.4)

    resultCache = {}
    for i, eachQuery in enumerate(queries):
        #print('Each query:', eachQuery)
        print('i:', i)
        #print('Each query type:', type(eachQuery))
        query =[]
        query.append(eachQuery)
        results = search(searcher, query, query_id_start=query_id_start)

        # Debug info
        #print(f"Number of results: {len(results)}")
        #print(f"Sample result: {list(results.items())[0] if results else 'No results'}")

        searchResult = list(results.items())[0]
        #print('print searchResult: ', searchResult)
        contextDocument = []
        for contentTuple in searchResult[1]:
            #print('contentTuple:', contentTuple)
            r2 = contentTuple.split(',')
            #print('r2s:', r2)
            #print('r21:', r2[1])
            r3=r2[1]
            #print(type(r3))
            r4 = r3.split(' : ')
            #print('r4:', r4[1])
            contextDocument.append(r4[1])

        #print(contextDocument)
        resultCache[i] = contextDocument

    # Save results
    with open("cacheResults_aws-service-logs.json", "w") as f:
        json.dump(resultCache, f)
        
def searchMatchingDocuments(query,llm,prompt ):
    '''
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant that identify if meaning of {input} and {second_sentence} is same. Just answer in yes or no",
        ),
        ("human", "{input}"),
    ]
    )
    '''

    chain = prompt | llm


    base_dir = f"data/logs/"
    dataset = "aws-service-logs"


    query_file = os.path.join(base_dir, f"{dataset}-queries.txt")


    # Load queries
    queries = load_queries(query_file)

    userQuery = query

    matchingTexts = ""
    print('Looking in cache memory if a similer search has been performed. \n \n')
    for i, eachQuery in enumerate(queries):
        #print(eachQuery)
        ai_msg= chain.invoke(
            {
                "second_sentence": eachQuery,
                "input": userQuery,
            }
        )
        #print(ai_msg)
        #print(ai_msg.content)
    
        if ai_msg.content == 'yes\n':
            #print(i)
            print('Found a match in Cache Memory, It will skip search and provide result from memory. \n \n')
            cacheResult = []
            with open("cacheResults_aws-service-logs.json", "r") as f:
                cacheResult=json.load(f)
            matchingTexts = cacheResult[str(i)]
            break

    foundInCache = 'y'        
    if len(matchingTexts) <= 0:
        foundInCache = 'n'
        querylist = []
        querylist.append(userQuery)
        print('Not found a match in Cache Memory. It is performing BM25 search to provide result and updating cache memory for subsequent search. \n \n')
        matchingTexts = mainSearch(dataset, querylist)
    
    print(f'Matching texts of query {userQuery} are  : \n \n {matchingTexts} \n')
    return foundInCache, matchingTexts



def initialize_session_state():
    #genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    if "messages" not in st.session_state:
        st.session_state.messages = []

def clear_chat_history():
    st.session_state.messages = []
    # Add a success message that will automatically disappear
    st.success("Chat history cleared!")
    #time.sleep(1)
    st.rerun()

def display_chat_history():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("API key not found")
        return False
                
    # Configure genai
    genai.configure(api_key=api_key)

    #genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    #print("API key:", os.environ["GOOGLE_API_KEY"])

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        google_api_key=api_key,
        timeout=30,
        max_retries=2
        # other params...
    )
    
    #print("LLM:", llm)
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that identify if meaning of {input} and {second_sentence} is same. Just answer in yes or no",
            ),
            ("human", "{input}"),
        ]
    )
    #print("Prompt template:", prompt_template)
    st.title("Interactive Chatbot")
    
    # Initialize session state
    initialize_session_state()
    
    # Create a sidebar with clear button and confirmation
    with st.sidebar:
        st.title("Chat Controls")
        if st.button("Clear Chat History", type="primary"):
            if len(st.session_state.messages) > 0:
                if st.sidebar.button("⚠️ Confirm Clear?", type="secondary"):
                    st.session_state.messages = []
                    st.success("Chat history cleared!")
                    #clear_chat_history()
            else:
                st.info("Chat is already empty!")
    
    # Display chat history
    display_chat_history()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        init_description = 'Hi I am CS410 search bot. I can search aws logs and provide you maching logs. Sample query is - "PUT /login 200 on 28/nov/2024"'
        message_placeholder.markdown(init_description + "▌")
        #st.session_state.messages.append({"role": "assistant", "content": init_description})
        message_placeholder = st.empty()
    
    # Chat input
    if prompt := st.chat_input("What query you want me to search?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simulate stream of response with a loading effect
            with st.spinner("Thinking..."):
                initResponse = 'Looking in cache memory if a similer search has been performed.'
                message_placeholder.markdown(initResponse + "▌")
                st.session_state.messages.append({"role": "assistant", "content": initResponse})
                message_placeholder = st.empty()
                foundInCache, response = get_chatbot_response(prompt,llm, prompt_template)
                if foundInCache == 'y':
                    search_response = 'Found a match in Cache Memory, I am skipping search and providing result from memory. \n \n' 
                else:
                    search_response = 'Not found a match in Cache Memory. I am performing BM25 search to provide result and updating cache memory for subsequent search. \n \n' 
                message_placeholder.markdown(search_response + "▌")
                st.session_state.messages.append({"role": "assistant", "content": search_response})
                message_placeholder = st.empty()
                subject_response = f'Matching texts of query {prompt} are  : \n \n'
                message_placeholder.markdown(subject_response + "▌")
                st.session_state.messages.append({"role": "assistant", "content": subject_response})
                message_placeholder = st.empty()
            
            for chunk in response.split():
                full_response += chunk + " "
                time.sleep(0.05)  # Add slight delay for effect
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})

def get_chatbot_response(prompt, llm, prompt_template):
    try:
        foundInCache, responseList = searchMatchingDocuments(prompt, llm, prompt_template)
        response = ''.join(str(eachDocument) for eachDocument in responseList)
        return foundInCache, response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

if __name__ == "__main__":
    loadCache(dataset)
    main()
