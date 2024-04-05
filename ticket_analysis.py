# ------------------------------------------------------------------------------
#	IMPORTS
# ------------------------------------------------------------------------------
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
# from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_mistralai.embeddings import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Investigate Re-Ranking Mechanism --- \\

from pathlib import Path
import textwrap

# ------------------------------------------------------------------------------
#	CODE
# ------------------------------------------------------------------------------
#	-- Load the Tickets
ticket_location = Path("/home/droog/.pyenv/versions/3.11.3/envs/langchain/main/zend_tickets")
ticket_source = DirectoryLoader(
	ticket_location,
	glob="**/*.txt",
	loader_cls=TextLoader,
	use_multithreading=True,
	show_progress=True )
tickets = ticket_source.load()

#	-- Chunk the Data
text_splitter = RecursiveCharacterTextSplitter()
split_tickets = text_splitter.split_documents(tickets)

#	-- Define embedding model and set up the vector store
embeddings = OllamaEmbeddings(model="mistral")
tickets_vector = FAISS.from_documents(split_tickets, embeddings)
tickets_retriever = tickets_vector.as_retriever()

#	-- Define the LLM
llm = Ollama(model="mistral")

#	-- Define the prompt
user_question = "What is the most recent Qwest ticket?"

prompt = ChatPromptTemplate.from_template(
"""Answer the following question based only on the provided context:
<context>
{context}
</context>

Question: {input}
""")

# Create a retrival chain to answer the questions
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(tickets_retriever, document_chain)

# Process the questions
response = retrieval_chain.invoke( {"input": user_question} )
answer = response["answer"]

print('-'*50)
print("\tZendesk Ticket RAG")
print('-'*20)
print(textwrap.fill(user_question, width=79))
print('-'*10)
print()
print(textwrap.fill(answer, width=79))
print()
print('-'*50)
