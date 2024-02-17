from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # tries to make chunks of size <= chunk_size or the minimum if all are > chunk_size
    chunk_overlap=100,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

# for doc in docs:
#     print(doc.page_content)
#     print("\n")

db = Chroma.from_documents(docs, embedding=embeddings, persist_directory="emb")


print(">>>>>>>>>>>>>>>Testing<<<<<<<<<<<<<<<")

# results = db.similarity_search_with_score(
#     "What is an interesting fact about the English language?",
#     k=3,  # default is 4
# )

# for (
#     result
# ) in (
#     results
# ):  # result is a tuple since we use similarity_search_with_score not similarity_search
#     print(result[1])  # search score
#     print(result[0].page_content)  # actual document
# results = db.similarity_search_with_score(
#     "What is an interesting fact about the English language?",
#     k=3,  # default is 4
# )

results = db.similarity_search(
    "What is an interesting fact about the English language?",
    k=3,  # default is 4
)

for result in results:
    print(result.page_content)
