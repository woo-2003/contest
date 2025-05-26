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

def get_relevant_documents(query: str, k: int = 4) -> List[Document]:
    """
    쿼리와 관련된 문서를 검색합니다.
    """
    if vectorstore is None:
        return []
    
    try:
        # 유사도 검색 수행
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        logger.error(f"Error retrieving documents: {e}")
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

def query_pdf_content(query: str, chat_history: List[Tuple[str, str]] = None) -> str:
    """
    PDF 내용에 대해 질문하고 답변을 반환합니다.
    """
    try:
        chain = create_rag_chain()
        if chain is None:
            return "PDF 데이터베이스가 초기화되지 않았습니다. PDF를 먼저 업로드해주세요."
        
        # 쿼리 실행
        result = chain({"question": query, "chat_history": chat_history or []})
        
        # 소스 문서 정보 추가
        sources = []
        for doc in result.get("source_documents", []):
            if "filename" in doc.metadata:
                sources.append(doc.metadata["filename"])
        
        # 답변과 소스 정보 결합
        answer = result["answer"]
        if sources:
            answer += f"\n\n참고한 문서: {', '.join(set(sources))}"
        
        return answer
        
    except Exception as e:
        logger.error(f"Error querying PDF content: {e}")
        return f"PDF 내용을 검색하는 중 오류가 발생했습니다: {str(e)}"

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