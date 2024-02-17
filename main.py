from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200,  # tries to make chunks of size <= chunk_size or the minimum if all are > chunk_size
    chunk_overlap=100,
)

loader = TextLoader("facts.txt")
docs = loader.load_and_split(text_splitter=text_splitter)

for doc in docs:
    print(doc.page_content)
    print("\n")