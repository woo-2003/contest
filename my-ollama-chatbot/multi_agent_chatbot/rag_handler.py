import os
from pathlib import Path
from typing import List, Tuple, Optional
import time
import tempfile
import shutil
import logging

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from .llm_config import embeddings, llm_coding # JS 변환에 코딩 LLM 사용
from .utils import extract_javascript_from_text, convert_js_to_python_code

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
PDF_STORAGE_PATH = str(BASE_DIR / "data" / "pdfs")

# 필요한 디렉토리 생성
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# 임베딩 모델 초기화
try:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
except ImportError as e:
    logger.error(f"Failed to initialize embeddings: {e}")
    logger.error("Please install required packages: pip install sentence-transformers torch transformers")
    raise
except Exception as e:
    logger.error(f"Unexpected error during embeddings initialization: {e}")
    raise

# 벡터 저장소 초기화
vectorstore = None

def initialize_chroma():
    """ChromaDB를 안전하게 초기화합니다."""
    max_retries = 3
    retry_delay = 2  # 초

    for attempt in range(max_retries):
        try:
            # 기존 데이터베이스 파일이 있다면 삭제
            if os.path.exists(CHROMA_DB_PATH):
                try:
                    shutil.rmtree(CHROMA_DB_PATH)
                except PermissionError:
                    print(f"데이터베이스 파일이 사용 중입니다. {retry_delay}초 후 재시도합니다...")
                    time.sleep(retry_delay)
                    continue

            os.makedirs(CHROMA_DB_PATH, exist_ok=True)
            
            # 새로운 ChromaDB 인스턴스 생성
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="rag_collection"
            )
            return vectorstore

        except Exception as e:
            print(f"ChromaDB 초기화 시도 {attempt + 1}/{max_retries} 실패: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise Exception("ChromaDB 초기화에 실패했습니다.")

# ChromaDB 초기화
vectorstore = initialize_chroma()

def process_and_embed_pdf(pdf_path: str) -> bool:
    """
    PDF 파일을 처리하고 임베딩하여 벡터 저장소에 저장합니다.
    """
    global vectorstore
    
    try:
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp()
        temp_pdf_path = os.path.join(temp_dir, "temp.pdf")
        
        # PDF 파일 복사
        shutil.copy2(pdf_path, temp_pdf_path)
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # PDF 로더 초기화
        loader = PyPDFLoader(temp_pdf_path)
        
        # PDF 텍스트 추출
        docs = loader.load()
        
        if not docs:
            logger.error(f"No content extracted from PDF: {pdf_path}")
            return False
        
        # 텍스트 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            logger.error(f"No text splits created from PDF: {pdf_path}")
            return False
        
        # 벡터 저장소 생성 또는 업데이트
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
        else:
            vectorstore.add_documents(splits)
        
        # 임시 파일 정리
        shutil.rmtree(temp_dir)
        
        logger.info(f"Successfully processed PDF: {pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        # 임시 파일 정리 시도
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass
        return False

def query_rag(query: str, k: int = 3) -> List:
    """
    RAG를 사용하여 쿼리에 대한 관련 문서를 검색합니다.
    """
    if vectorstore is None:
        logger.warning("Vector store is not initialized. No documents have been processed yet.")
        return []
    
    try:
        return vectorstore.similarity_search(query, k=k)
    except Exception as e:
        logger.error(f"Error querying RAG: {str(e)}")
        return []

def get_rag_retriever():
    """
    RAG 검색기를 반환합니다.
    """
    if vectorstore is None:
        logger.warning("Vector store is not initialized. No documents have been processed yet.")
        return None
    
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

def list_available_collections():
    """ 사용 가능한 ChromaDB 컬렉션 목록 (디버깅용) """
    client = Chroma(persist_directory=CHROMA_DB_PATH)
    chromadb_client = client._client
    collection = chromadb_client.list_collections()
    return [col.name for col in collection]

# 초기화 시 기존 DB 로드 확인
print(f"ChromaDB initialized. Available collections: {list_available_collections()}")
if not any(col == "rag_collection" for col in list_available_collections()):
    print("Warning: 'rag_collection' not found. PDFs might need to be re-uploaded.")