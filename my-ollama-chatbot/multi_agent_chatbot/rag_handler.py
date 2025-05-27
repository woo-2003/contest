import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import time
import tempfile
import shutil
import logging
from datetime import datetime
import chromadb
import json

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from .llm_config import embeddings, llm_general

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 데이터 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
PDF_STORAGE_PATH = str(BASE_DIR / "data" / "pdfs")
PDF_METADATA_PATH = str(BASE_DIR / "data" / "pdf_metadata.json")
PDF_INDEX_PATH = str(BASE_DIR / "data" / "pdf_index.json")

# 필요한 디렉토리 생성
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# PDF 메타데이터 관리
pdf_metadata = {}
pdf_index = {}  # PDF 파일 경로와 ID 매핑

def save_pdf_metadata():
    """PDF 메타데이터를 파일에 저장합니다."""
    with open(PDF_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_metadata, f, ensure_ascii=False, indent=2)

def load_pdf_metadata():
    """PDF 메타데이터를 파일에서 로드합니다."""
    global pdf_metadata
    if os.path.exists(PDF_METADATA_PATH):
        with open(PDF_METADATA_PATH, 'r', encoding='utf-8') as f:
            pdf_metadata = json.load(f)

def save_pdf_index():
    """PDF 인덱스를 파일에 저장합니다."""
    with open(PDF_INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_index, f, ensure_ascii=False, indent=2)

def load_pdf_index():
    """PDF 인덱스를 파일에서 로드합니다."""
    global pdf_index
    if os.path.exists(PDF_INDEX_PATH):
        with open(PDF_INDEX_PATH, 'r', encoding='utf-8') as f:
            pdf_index = json.load(f)

# 메타데이터와 인덱스 로드
load_pdf_metadata()
load_pdf_index()

def initialize_chroma():
    """ChromaDB를 초기화하고 기존 데이터를 로드합니다."""
    try:
        # ChromaDB 클라이언트 초기화
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        # 기존 컬렉션이 없으면 새로 생성
        if "rag_collection" not in [col.name for col in client.list_collections()]:
            logger.info("새로운 rag_collection 생성")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="rag_collection"
            )
        else:
            logger.info("기존 rag_collection 로드")
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="rag_collection"
            )
        
        return vectorstore
    except Exception as e:
        logger.error(f"ChromaDB 초기화 실패: {e}")
        return None

# ChromaDB 초기화
vectorstore = initialize_chroma()

def process_and_embed_pdf(pdf_path: str) -> bool:
    """
    PDF 파일을 처리하고 임베딩하여 벡터 저장소에 저장합니다.
    """
    global vectorstore, pdf_index
    
    try:
        # PDF 파일명 추출
        pdf_filename = os.path.basename(pdf_path)
        pdf_id = str(int(time.time()))  # 고유 ID 생성
        
        # 이미 처리된 PDF인지 확인
        if pdf_path in pdf_index:
            logger.info(f"PDF already processed: {pdf_path}")
            return True
        
        # PDF 메타데이터 생성
        pdf_metadata[pdf_id] = {
            "filename": pdf_filename,
            "upload_time": datetime.now().isoformat(),
            "status": "processing"
        }
        save_pdf_metadata()
        
        # PDF 파일을 영구 저장소로 복사
        permanent_path = os.path.join(PDF_STORAGE_PATH, f"{pdf_id}_{pdf_filename}")
        shutil.copy2(pdf_path, permanent_path)
        
        # PDF 인덱스에 추가
        pdf_index[pdf_path] = {
            "id": pdf_id,
            "permanent_path": permanent_path,
            "filename": pdf_filename
        }
        save_pdf_index()
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # PyPDFLoader 사용
        loader = PyPDFLoader(permanent_path)
        
        # PDF 텍스트 추출
        docs = loader.load()
        
        if not docs:
            logger.error(f"No content extracted from PDF: {pdf_path}")
            pdf_metadata[pdf_id]["status"] = "failed"
            save_pdf_metadata()
            return False
        
        # 텍스트 분할 최적화
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        splits = text_splitter.split_documents(docs)
        
        if not splits:
            logger.error(f"No text splits created from PDF: {pdf_path}")
            pdf_metadata[pdf_id]["status"] = "failed"
            save_pdf_metadata()
            return False
        
        # 문서에 메타데이터 추가
        for split in splits:
            split.metadata.update({
                "pdf_id": pdf_id,
                "filename": pdf_filename,
                "chunk_index": len(splits)
            })
        
        # 벡터 저장소에 추가
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=splits,
                embedding=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="rag_collection"
            )
        else:
            vectorstore.add_documents(splits)
            vectorstore.persist()
        
        # 메타데이터 업데이트
        pdf_metadata[pdf_id].update({
            "status": "completed",
            "chunks": len(splits),
            "total_chars": sum(len(doc.page_content) for doc in splits)
        })
        save_pdf_metadata()
        
        logger.info(f"Successfully processed PDF: {pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
        if pdf_id in pdf_metadata:
            pdf_metadata[pdf_id]["status"] = "failed"
            pdf_metadata[pdf_id]["error"] = str(e)
            save_pdf_metadata()
        return False

def get_relevant_documents(query: str, k: int = 3) -> List[Document]:
    """
    쿼리와 관련된 문서를 검색합니다.
    
    Args:
        query (str): 검색 쿼리
        k (int): 반환할 문서 수
        
    Returns:
        List[Document]: 관련 문서 리스트
    """
    try:
        # 쿼리 전처리
        processed_query = query.strip().lower()
        
        # ChromaDB 검색
        results = vectorstore.query(
            query_texts=[processed_query],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results["documents"][0]:
            logger.warning(f"검색 결과 없음: {query}")
            return []
        
        # 결과를 Document 객체로 변환
        documents = []
        for doc, metadata, distance in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ):
            # 관련성 점수 계산 (거리를 0-1 사이의 점수로 변환)
            relevance_score = 1.0 - min(distance, 1.0)
            
            # 메타데이터에 관련성 점수 추가
            metadata["relevance_score"] = relevance_score
            
            # Document 객체 생성
            document = Document(
                page_content=doc,
                metadata=metadata
            )
            documents.append(document)
        
        # 관련성 점수로 정렬
        documents.sort(key=lambda x: x.metadata.get("relevance_score", 0), reverse=True)
        
        logger.info(f"검색 완료: {len(documents)}개 문서 발견")
        return documents
        
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {str(e)}", exc_info=True)
        return []

def create_rag_chain():
    """
    RAG 체인을 생성합니다.
    """
    if vectorstore is None:
        return None
    
    try:
        # 대화 메모리 설정
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # RAG 체인 생성
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_general,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4}
            ),
            memory=memory,
            return_source_documents=True
        )
        
        return chain
    except Exception as e:
        logger.error(f"Error creating RAG chain: {e}")
        return None

def query_pdf_content(query: str, k: int = 3) -> str:
    """
    PDF 내용을 검색하고 관련 컨텍스트를 생성합니다.
    
    Args:
        query (str): 검색 쿼리
        k (int): 반환할 문서 수
        
    Returns:
        str: 검색 결과 컨텍스트
    """
    try:
        # 관련 문서 검색
        relevant_docs = get_relevant_documents(query, k=k)
        
        if not relevant_docs:
            return "관련 정보를 찾을 수 없습니다."
        
        # 컨텍스트 생성
        context_parts = []
        for doc in relevant_docs:
            # 관련성 점수가 0.5 이상인 경우만 포함
            if doc.metadata.get("relevance_score", 0) >= 0.5:
                source = doc.metadata.get("source", "알 수 없는 출처")
                page = doc.metadata.get("page", "알 수 없는 페이지")
                context_parts.append(f"[출처: {source}, 페이지: {page}]\n{doc.page_content}")
        
        if not context_parts:
            return "관련성 높은 정보를 찾을 수 없습니다."
        
        return "\n\n".join(context_parts)
        
    except Exception as e:
        logger.error(f"PDF 내용 검색 중 오류 발생: {str(e)}", exc_info=True)
        return "문서 검색 중 오류가 발생했습니다."

def list_available_collections() -> List[str]:
    """
    사용 가능한 ChromaDB 컬렉션 목록을 반환합니다.
    """
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return []

def get_processed_pdfs() -> List[Dict]:
    """
    처리된 PDF 파일 목록을 반환합니다.
    """
    return [
        {
            "filename": info["filename"],
            "id": info["id"],
            "status": pdf_metadata[info["id"]]["status"]
        }
        for info in pdf_index.values()
    ]

# 초기화 시 기존 DB 로드 확인
print(f"ChromaDB initialized. Available collections: {list_available_collections()}")
if not any(col == "rag_collection" for col in list_available_collections()):
    print("Warning: 'rag_collection' not found. PDFs might need to be re-uploaded.")