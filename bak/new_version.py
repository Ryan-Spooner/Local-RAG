# ------------------------------------------------------------------------------
#	IMPORTS
# ------------------------------------------------------------------------------
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Investigate Re-Ranking Mechanism

import textwrap

# ------------------------------------------------------------------------------
#	CODE
# ------------------------------------------------------------------------------
#	-- Load the Data
web_source = "https://docs.smith.langchain.com/user_guide"
local_source = "sample.txt"
web_docs = WebBaseLoader(web_source).load()
local_docs = TextLoader(local_source).load()

#	-- Chunk the Data
text_splitter = RecursiveCharacterTextSplitter()
web_docs_split = text_splitter.split_documents(web_docs)
local_docs_split = text_splitter.split_documents(local_docs)

#	-- Define embedding model and set up the vector store
embeddings = OllamaEmbeddings(model="mistral")
vector_web = FAISS.from_documents(web_docs_split, embeddings)
vector_local = FAISS.from_documents(local_docs_split, embeddings)
retriever_web = vector_web.as_retriever()
retriever_local = vector_local.as_retriever()

#	-- Define the LLM
llm = Ollama(model="mistral")

#	-- Define the prompt
user_input_web = "How can langsmith help with testing?"
user_input_local = "My name is Ryan Spooner. What is my middle name?"

prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}
""")

# Create a retrival chain to answer the questions
document_chain = create_stuff_documents_chain(llm, prompt)
web_retrieval_chain = create_retrieval_chain(retriever_web, document_chain)
local_retrieval_chain = create_retrieval_chain(retriever_local, document_chain)

# Process the questions
web_response = web_retrieval_chain.invoke( {"input": user_input_web} )
local_response = local_retrieval_chain.invoke( {"input": user_input_local} )
web_answer = web_response["answer"]
local_answer = local_response["answer"]

print('-'*50)
print("\tWEB DATA RAG")
print('-'*20)
print(textwrap.fill(user_input_web, width=79))
print('-'*10)
print()
print(textwrap.fill(web_answer, width=79))
print()
print('-'*50)
print("\tLOCAL DATA RAG")
print('-'*20)
print(textwrap.fill(user_input_local, width=79))
print('-'*10)
print()
print(textwrap.fill(local_answer, width=79))
