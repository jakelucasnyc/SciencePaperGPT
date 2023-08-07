from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from pprint import pprint
import logging
import time
from appdirs import user_cache_dir, user_data_dir
import json
import shutil

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

data_dir = user_data_dir('SciencePaperGPT', 'BioGear Labs')
os.makedirs(data_dir, exist_ok=True)

def clear_cached_papers(cache_dir:str):
    if os.path.isfile(os.path.join(cache_dir, 'papers.json')):
        os.remove(os.path.join(cache_dir, 'papers.json'))
    if os.path.isdir(os.path.join(data_dir, 'papers')):
        shutil.rmtree(os.path.join(data_dir, 'papers'))
    

def create_db_from_pdf(pdf_link:str, embeddings:OpenAIEmbeddings):
    if not os.path.isfile(pdf_link):
        raise FileNotFoundError(f'PDF provided {pdf_link} does not exist.')
    loader = UnstructuredPDFLoader(pdf_link)

    logger.info(f'Loading PDF {pdf_link}...')
    start = time.perf_counter()
    data = loader.load()
    elapsed = time.perf_counter() - start
    logger.info(f'Loaded PDF {pdf_link} in {elapsed: .2f}s')

    # pprint(data[0].page_content)

    logger.info('Splitting document into chunks...')
    start = time.perf_counter()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(data)
    elapsed = time.perf_counter() - start
    logger.info(f'Document split in {elapsed: .2f}s')

    logger.info('Creating FAISS Database...')
    start = time.perf_counter()
    faiss_db = FAISS.from_documents(docs, embeddings)
    elapsed = time.perf_counter() - start
    logger.info(f'FAISS Database created in {elapsed: .2f}s')
    return faiss_db 

def save_faiss_db(db:FAISS, cache:dict, pdf_link:str):
    if cache:
        last_file = list(cache.keys())[-1]
        last_num = int(last_file[-5:])
        new_num = last_num+1
    else:
        new_num = 1
    new_folder = f'paper{str(new_num).zfill(5)}'
    cache.update({new_folder: pdf_link})
    new_path = os.path.join(data_dir, 'papers', new_folder)
    logger.info(f'Saving FAISS Database:{new_path}...')
    try:
        db.save_local(new_path)
    except Exception as e:
        cache.pop(new_folder)
        raise e
    logger.info(f'FAISS Database saved')
    # pprint(cache)
    return cache

def load_faiss_db(pdf_link:str, cache:dict, embeddings:OpenAIEmbeddings):
    paper_id = list(cache.keys())[list(cache.values()).index(pdf_link)]
    db_path = os.path.join(data_dir, 'papers', paper_id)
    logger.info(f'Loading FAISS Database:{db_path}...')
    db = FAISS.load_local(db_path, embeddings)
    logger.info(f'FAISS Database loaded')
    return db

def retrieve_pdf_data(pdf_link:str, embeddings:OpenAIEmbeddings, cache_dir:str):
    with open(os.path.join(cache_dir, 'papers.json'), 'r') as f:
        cache = json.load(f)
    if pdf_link in cache.values():
        faiss_db = load_faiss_db(pdf_link, cache, embeddings)
    else:
        faiss_db = create_db_from_pdf(pdf_link, embeddings)
        cache = save_faiss_db(faiss_db, cache, pdf_link)

    with open(os.path.join(cache_dir, 'papers.json'), 'w') as f:
        json.dump(cache, f)

    return faiss_db
