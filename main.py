from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()
db = Chroma(embedding_function=embeddings, persist_directory="emb")
retriever = db.as_retriever()
chat = ChatOpenAI()

chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.invoke("What is an interesting fact about the English language?")

print(result)
