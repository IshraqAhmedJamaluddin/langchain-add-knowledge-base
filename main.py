from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from deduplication_retriever import DeDuplicationRetriever
from dotenv import load_dotenv

# import langchain
# langchain.debug = True

load_dotenv()

embeddings = OpenAIEmbeddings()
db = Chroma(embedding_function=embeddings, persist_directory="emb")
retriever = DeDuplicationRetriever(embeddings=embeddings, chroma=db)
chat = ChatOpenAI()

chain = RetrievalQA.from_chain_type(llm=chat, retriever=retriever, chain_type="stuff")

result = chain.invoke("What is an interesting fact about the English language?")

print(result)
