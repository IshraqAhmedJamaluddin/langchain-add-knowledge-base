# langchain-add-knowledge-base

This project answers your questions based on a facts.txt file as a knowledge base

## Setup and Running

- add a `.env` file to your base directory that includes your `OPENAI_API_KEY` e.g:

  ```
  OPENAI_API_KEY=[YOUR KEY HERE]
  ```

- to make an environment use `python -m venv env`, then if you're on mac or linux use `source env/bin/activate` or on windows use the command corresponding to your terminal type.

- to install required packages run `pip install -r requirements.txt`

- to load the data into chromadb run `python loader.py` only once

- to run just type `python main.py`
