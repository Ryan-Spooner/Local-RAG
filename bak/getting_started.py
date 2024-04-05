# ------------------------------------------------------------------------------
# --	IMPORTS
# ------------------------------------------------------------------------------
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
#from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Investigate Re-Ranking Mechanism

import textwrap

# ------------------------------------------------------------------------------
# --	CODE
# ------------------------------------------------------------------------------
llm = Ollama(model="mistral", temerature=0.1)
embeddings = OllamaEmbeddings(model="mistral")
loader = WebBaseLoader("https://docs.smith.langchain.com/user_guide")

docs = loader.load()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)

user_input = "How can langsmith help with testing?"
model_role_definition = "You are a world class technical documentation writer."

# prompt = ChatPromptTemplate.from_messages([
# 	("system", model_role_definition),
# 	("user", "{input}")
# ])
prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}""")

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = vector.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
output_parser = StrOutputParser()

chain = prompt | llm | output_parser

response = retrieval_chain.invoke( {"input": user_input} )
answer = response["answer"]

print( textwrap.fill(user_input, width=79) )
print('-'*15)
print()
print( textwrap.fill(answer, width=80) )
