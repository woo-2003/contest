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
import hashlib
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from langchain_community.document_loaders import PyPDFLoader, PDFMinerLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from .llm_config import embeddings, llm_general

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 데이터 디렉토리 설정
BASE_DIR = Path(__file__).parent.parent.parent
CHROMA_DB_PATH = str(BASE_DIR / "data" / "chroma_db")
PDF_STORAGE_PATH = str(BASE_DIR / "data" / "pdfs")
PDF_METADATA_PATH = str(BASE_DIR / "data" / "pdf_metadata.json")
PDF_INDEX_PATH = str(BASE_DIR / "data" / "pdf_index.json")
PDF_HASH_PATH = str(BASE_DIR / "data" / "pdf_hashes.json")

# 필요한 디렉토리 생성
os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# PDF 메타데이터 관리
pdf_metadata = {}
pdf_index = {}  # PDF 파일 경로와 ID 매핑
pdf_hashes = {}  # PDF 파일 해시값 저장

def calculate_file_hash(file_path: str) -> str:
    """파일의 MD5 해시를 계산합니다."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

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

def save_pdf_hashes():
    """PDF 해시값을 파일에 저장합니다."""
    with open(PDF_HASH_PATH, 'w', encoding='utf-8') as f:
        json.dump(pdf_hashes, f, ensure_ascii=False, indent=2)

def load_pdf_hashes():
    """PDF 해시값을 파일에서 로드합니다."""
    global pdf_hashes
    if os.path.exists(PDF_HASH_PATH):
        with open(PDF_HASH_PATH, 'r', encoding='utf-8') as f:
            pdf_hashes = json.load(f)

# 메타데이터, 인덱스, 해시값 로드
load_pdf_metadata()
load_pdf_index()
load_pdf_hashes()

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

def extract_text_with_ocr(pdf_path: str) -> List[Document]:
    """OCR을 사용하여 PDF에서 텍스트를 추출합니다."""
    try:
        # PDF 열기
        pdf_document = fitz.open(pdf_path)
        documents = []
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # 페이지를 이미지로 변환
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # OCR 수행
            text = pytesseract.image_to_string(img, lang='kor+eng')
            
            if text.strip():
                # Document 객체 생성
                doc = Document(
                    page_content=text,
                    metadata={
                        "page": page_num + 1,
                        "source": pdf_path
                    }
                )
                documents.append(doc)
        
        pdf_document.close()
        return documents
    except Exception as e:
        logger.error(f"OCR 처리 중 오류 발생: {str(e)}")
        return []

def validate_pdf(pdf_path: str) -> Tuple[bool, str]:
    """
    PDF 파일의 유효성을 검사합니다.
    
    Returns:
        Tuple[bool, str]: (유효성 여부, 오류 메시지)
    """
    try:
        # 파일 존재 확인
        if not os.path.exists(pdf_path):
            return False, "파일이 존재하지 않습니다."
            
        # 파일 크기 확인
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            return False, "파일이 비어있습니다."
        if file_size < 100:  # 최소 PDF 크기
            return False, f"파일 크기가 너무 작습니다 ({file_size} bytes). 유효한 PDF 파일이 아닐 수 있습니다."
            
        # PDF 헤더 확인
        with open(pdf_path, 'rb') as f:
            header = f.read(5)
            if header != b'%PDF-':
                return False, "유효한 PDF 파일이 아닙니다. PDF 헤더가 올바르지 않습니다."
                
        # PyMuPDF로 PDF 열기 시도
        try:
            doc = fitz.open(pdf_path)
            if doc.is_encrypted:
                return False, "암호화된 PDF 파일입니다. 암호를 제거한 후 다시 시도해주세요."
            if doc.page_count == 0:
                return False, "PDF 파일에 페이지가 없습니다."
            doc.close()
        except Exception as e:
            return False, f"PDF 파일을 열 수 없습니다: {str(e)}"
            
        return True, "유효한 PDF 파일입니다."
        
    except Exception as e:
        return False, f"파일 검증 중 오류 발생: {str(e)}"

def process_and_embed_pdf(pdf_path: str) -> bool:
    """
    PDF 파일을 처리하고 임베딩하여 벡터 저장소에 저장합니다.
    """
    global vectorstore, pdf_index
    
    try:
        # PDF 파일명 추출
        pdf_filename = os.path.basename(pdf_path)
        
        # PDF 파일 검증
        is_valid, error_message = validate_pdf(pdf_path)
        if not is_valid:
            logger.error(f"PDF 파일 검증 실패: {pdf_filename} - {error_message}")
            return False
        
        # 파일 해시 계산
        file_hash = calculate_file_hash(pdf_path)
        
        # 이미 처리된 PDF인지 해시값으로 확인
        if file_hash in pdf_hashes:
            logger.info(f"PDF already processed (hash match): {pdf_filename}")
            return True
        
        pdf_id = str(int(time.time()))  # 고유 ID 생성
        
        # PDF 메타데이터 생성
        pdf_metadata[pdf_id] = {
            "filename": pdf_filename,
            "upload_time": datetime.now().isoformat(),
            "status": "processing",
            "file_hash": file_hash
        }
        save_pdf_metadata()
        
        # PDF 파일을 영구 저장소로 복사
        permanent_path = os.path.join(PDF_STORAGE_PATH, f"{pdf_id}_{pdf_filename}")
        shutil.copy2(pdf_path, permanent_path)
        
        # PDF 인덱스에 추가
        pdf_index[pdf_path] = {
            "id": pdf_id,
            "permanent_path": permanent_path,
            "filename": pdf_filename,
            "file_hash": file_hash
        }
        save_pdf_index()
        
        # 해시값 저장
        pdf_hashes[file_hash] = {
            "pdf_id": pdf_id,
            "filename": pdf_filename,
            "permanent_path": permanent_path
        }
        save_pdf_hashes()
        
        logger.info(f"Processing PDF: {pdf_path}")
        
        # PDF 로더 시도
        docs = None
        loaders = [
            (PyPDFLoader, "PyPDFLoader"),
            (PDFMinerLoader, "PDFMinerLoader")
        ]
        
        # 일반 PDF 로더 시도
        for loader_class, loader_name in loaders:
            try:
                logger.info(f"Trying to load PDF with {loader_name}")
                loader = loader_class(permanent_path)
                docs = loader.load()
                if docs and any(doc.page_content.strip() for doc in docs):
                    logger.info(f"Successfully loaded PDF with {loader_name}")
                    break
            except Exception as e:
                logger.warning(f"Failed to load PDF with {loader_name}: {str(e)}")
                continue
        
        # 일반 로더가 실패한 경우 OCR 시도
        if not docs or not any(doc.page_content.strip() for doc in docs):
            logger.info("Attempting OCR processing...")
            docs = extract_text_with_ocr(permanent_path)
            if docs and any(doc.page_content.strip() for doc in docs):
                logger.info("Successfully processed PDF with OCR")
            else:
                logger.error(f"Failed to process PDF with OCR: {pdf_path}")
                pdf_metadata[pdf_id]["status"] = "failed"
                pdf_metadata[pdf_id]["error"] = "모든 PDF 처리 방법이 실패했습니다."
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
            pdf_metadata[pdf_id]["error"] = "텍스트 분할에 실패했습니다."
            save_pdf_metadata()
            return False
        
        # 문서에 메타데이터 추가
        for i, split in enumerate(splits):
            split.metadata.update({
                "pdf_id": pdf_id,
                "filename": pdf_filename,
                "chunk_index": i,
                "file_hash": file_hash,
                "processing_method": "ocr" if not docs else "standard"
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
            "total_chars": sum(len(doc.page_content) for doc in splits),
            "processing_method": "ocr" if not docs else "standard"
        })
        save_pdf_metadata()
        
        # 데이터 지속성 검증
        verify_data_persistence()
        
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
    """쿼리와 관련된 문서를 검색합니다."""
    if vectorstore is None:
        logger.error("벡터 저장소가 초기화되지 않았습니다.")
        return []
    
    try:
        # 일반적인 질문인 경우 모든 문서 반환
        if any(keyword in query.lower() for keyword in ["모든 내용", "전체 내용", "학습한 내용", "pdf 내용"]):
            # 모든 문서 가져오기
            results = vectorstore.get()
            if results and results["documents"]:
                logger.info(f"전체 문서 수: {len(results['documents'])}")
                return [Document(page_content=doc, metadata=meta) 
                       for doc, meta in zip(results["documents"], results["metadatas"])]
            logger.warning("저장된 문서가 없습니다.")
            return []
        
        # 특정 질문에 대한 유사도 검색
        docs = vectorstore.similarity_search_with_score(query, k=k)
        
        # 유사도 점수가 0.8 이상인 문서만 필터링 (임계값 상향 조정)
        filtered_docs = [
            doc for doc, score in docs 
            if score <= 0.8  # 유사도 점수가 낮을수록 더 유사함
        ]
        
        if not filtered_docs:
            logger.warning(f"유사도가 높은 문서를 찾을 수 없습니다: {query}")
            return []
            
        return filtered_docs
    except Exception as e:
        logger.error(f"문서 검색 중 오류 발생: {str(e)}")
        return []

def create_rag_chain():
    """RAG 체인을 생성합니다."""
    if vectorstore is None:
        return None
    
    try:
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.7  # 유사도 임계값 설정
            }
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm_general,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=True  # 디버깅을 위한 상세 로그 활성화
        )
        
        return chain
    except Exception as e:
        logger.error(f"RAG 체인 생성 중 오류 발생: {str(e)}")
        return None

def query_pdf_content(query: str, k: int = 3) -> str:
    """PDF 내용을 기반으로 쿼리에 답변합니다."""
    try:
        # 관련 문서 검색
        relevant_docs = get_relevant_documents(query, k=k)
        
        if not relevant_docs:
            return "죄송합니다. PDF에서 관련 정보를 찾을 수 없습니다. 다른 질문을 해보시거나, PDF가 제대로 업로드되었는지 확인해주세요."
        
        # 일반적인 질문인 경우 모든 내용 요약
        if any(keyword in query.lower() for keyword in ["모든 내용", "전체 내용", "학습한 내용", "pdf 내용"]):
            all_content = "\n\n".join([doc.page_content for doc in relevant_docs])
            return f"학습된 PDF의 전체 내용입니다:\n\n{all_content}"
        
        # RAG 체인 생성 및 실행
        chain = create_rag_chain()
        if chain is None:
            return "PDF 데이터베이스가 초기화되지 않았습니다."
        
        # 쿼리 실행
        response = chain({"question": query})
        
        # 소스 문서 정보 수집
        sources = []
        for doc in response.get("source_documents", []):
            source_info = {
                "filename": doc.metadata.get("filename", "알 수 없음"),
                "page": doc.metadata.get("page", "알 수 없음"),
                "processing_method": doc.metadata.get("processing_method", "standard")
            }
            sources.append(source_info)
        
        # 답변과 소스 정보 결합
        answer = response["answer"]
        if sources:
            source_text = "\n\n참고한 문서:\n"
            for source in sources:
                source_text += f"- {source['filename']} (페이지: {source['page']}, 처리방법: {source['processing_method']})\n"
            answer += source_text
        
        return answer
    except Exception as e:
        logger.error(f"PDF 내용 쿼리 중 오류 발생: {str(e)}")
        return f"오류가 발생했습니다: {str(e)}"

def verify_pdf_content(pdf_id: str) -> Dict:
    """PDF 내용의 학습 상태를 확인합니다."""
    try:
        if pdf_id not in pdf_metadata:
            return {
                "status": "error",
                "message": "PDF를 찾을 수 없습니다."
            }
        
        pdf_info = pdf_metadata[pdf_id]
        
        # 벡터 저장소에서 해당 PDF의 문서 검색
        if vectorstore is None:
            return {
                "status": "error",
                "message": "벡터 저장소가 초기화되지 않았습니다."
            }
        
        # PDF ID로 문서 검색
        results = vectorstore.get(
            where={"pdf_id": pdf_id}
        )
        
        if not results or not results["documents"]:
            return {
                "status": "error",
                "message": "PDF 내용을 찾을 수 없습니다."
            }
        
        return {
            "status": "success",
            "filename": pdf_info["filename"],
            "chunks": pdf_info.get("chunks", 0),
            "total_chars": pdf_info.get("total_chars", 0),
            "processing_method": pdf_info.get("processing_method", "standard"),
            "document_count": len(results["documents"]),
            "sample_text": results["documents"][0][:200] + "..." if results["documents"] else ""
        }
    except Exception as e:
        logger.error(f"Error verifying PDF content: {str(e)}")
        return {
            "status": "error",
            "message": f"확인 중 오류 발생: {str(e)}"
        }

def list_available_collections() -> List[str]:
    """사용 가능한 컬렉션 목록을 반환합니다."""
    try:
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        return [col.name for col in client.list_collections()]
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        return []

def get_processed_pdfs() -> List[Dict]:
    """처리된 PDF 파일 목록을 반환합니다."""
    return [
        {
            "id": pdf_id,
            "filename": data["filename"],
            "status": data["status"],
            "upload_time": data["upload_time"],
            "chunks": data.get("chunks", 0),
            "total_chars": data.get("total_chars", 0)
        }
        for pdf_id, data in pdf_metadata.items()
    ]

def cleanup_old_pdfs(days: int = 30):
    """지정된 기간 이상 된 PDF 파일을 정리합니다."""
    try:
        current_time = datetime.now()
        to_delete = []
        
        for pdf_id, data in pdf_metadata.items():
            upload_time = datetime.fromisoformat(data["upload_time"])
            if (current_time - upload_time).days > days:
                to_delete.append(pdf_id)
        
        for pdf_id in to_delete:
            # 메타데이터에서 제거
            if pdf_id in pdf_metadata:
                del pdf_metadata[pdf_id]
            
            # 인덱스에서 제거
            for path, index_data in list(pdf_index.items()):
                if index_data["id"] == pdf_id:
                    del pdf_index[path]
            
            # 해시값에서 제거
            for hash_value, hash_data in list(pdf_hashes.items()):
                if hash_data["pdf_id"] == pdf_id:
                    del pdf_hashes[hash_value]
            
            # 파일 삭제
            permanent_path = os.path.join(PDF_STORAGE_PATH, f"{pdf_id}_{data['filename']}")
            if os.path.exists(permanent_path):
                os.remove(permanent_path)
        
        # 변경사항 저장
        save_pdf_metadata()
        save_pdf_index()
        save_pdf_hashes()
        
        logger.info(f"Cleaned up {len(to_delete)} old PDF files")
        return len(to_delete)
    except Exception as e:
        logger.error(f"Error cleaning up old PDFs: {str(e)}")
        return 0

def reset_pdf_database():
    """PDF 학습 데이터를 완전히 초기화합니다."""
    try:
        global vectorstore, pdf_metadata, pdf_index, pdf_hashes
        
        logger.info("데이터베이스 초기화 시작...")
        
        # 초기화 전 상태 확인
        logger.info("초기화 전 상태:")
        logger.info(f"ChromaDB 경로 존재: {os.path.exists(CHROMA_DB_PATH)}")
        logger.info(f"PDF 저장소 경로 존재: {os.path.exists(PDF_STORAGE_PATH)}")
        logger.info(f"PDF 메타데이터 파일 존재: {os.path.exists(PDF_METADATA_PATH)}")
        
        # ChromaDB 초기화
        if os.path.exists(CHROMA_DB_PATH):
            logger.info(f"ChromaDB 디렉토리 삭제: {CHROMA_DB_PATH}")
            shutil.rmtree(CHROMA_DB_PATH)
            logger.info("ChromaDB 디렉토리 삭제 완료")
        
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        logger.info("ChromaDB 디렉토리 재생성 완료")
        
        # PDF 저장소 초기화
        if os.path.exists(PDF_STORAGE_PATH):
            logger.info(f"PDF 저장소 디렉토리 삭제: {PDF_STORAGE_PATH}")
            shutil.rmtree(PDF_STORAGE_PATH)
            logger.info("PDF 저장소 디렉토리 삭제 완료")
        
        os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
        logger.info("PDF 저장소 디렉토리 재생성 완료")
        
        # 메타데이터 파일 초기화
        for file_path in [PDF_METADATA_PATH, PDF_INDEX_PATH, PDF_HASH_PATH]:
            if os.path.exists(file_path):
                logger.info(f"메타데이터 파일 삭제: {file_path}")
                os.remove(file_path)
                logger.info(f"메타데이터 파일 삭제 완료: {file_path}")
        
        # 메모리 변수 초기화
        pdf_metadata = {}
        pdf_index = {}
        pdf_hashes = {}
        logger.info("메모리 변수 초기화 완료")
        
        # ChromaDB 재초기화
        logger.info("ChromaDB 재초기화 시작...")
        vectorstore = initialize_chroma()
        if vectorstore is None:
            raise Exception("ChromaDB 재초기화 실패")
        logger.info("ChromaDB 재초기화 완료")
        
        # 초기화 후 상태 확인
        logger.info("초기화 후 상태:")
        logger.info(f"ChromaDB 경로 존재: {os.path.exists(CHROMA_DB_PATH)}")
        logger.info(f"PDF 저장소 경로 존재: {os.path.exists(PDF_STORAGE_PATH)}")
        logger.info(f"PDF 메타데이터 파일 존재: {os.path.exists(PDF_METADATA_PATH)}")
        
        # 상태 정보 출력
        status = get_database_status()
        logger.info(f"최종 데이터베이스 상태: {status}")
        
        logger.info("PDF 데이터베이스 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"데이터베이스 초기화 중 오류 발생: {str(e)}")
        return False

def get_database_status() -> Dict:
    """현재 데이터베이스 상태를 반환합니다."""
    try:
        # ChromaDB 상태 확인
        chroma_size = 0
        if os.path.exists(CHROMA_DB_PATH):
            for dirpath, dirnames, filenames in os.walk(CHROMA_DB_PATH):
                for filename in filenames:
                    file_path = os.path.join(dirpath, filename)
                    chroma_size += os.path.getsize(file_path)

        # PDF 저장소 상태 확인
        pdf_files = []
        pdf_size = 0
        if os.path.exists(PDF_STORAGE_PATH):
            pdf_files = [f for f in os.listdir(PDF_STORAGE_PATH) if f.endswith('.pdf')]
            for pdf_file in pdf_files:
                file_path = os.path.join(PDF_STORAGE_PATH, pdf_file)
                pdf_size += os.path.getsize(file_path)

        # 메타데이터 파일 상태 확인
        metadata_exists = os.path.exists(PDF_METADATA_PATH)
        metadata_count = 0
        if metadata_exists:
            try:
                with open(PDF_METADATA_PATH, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                    metadata_count = len(metadata)
            except:
                metadata_count = 0

        status = {
            "chroma_db": {
                "exists": os.path.exists(CHROMA_DB_PATH),
                "size": chroma_size
            },
            "pdf_storage": {
                "exists": os.path.exists(PDF_STORAGE_PATH),
                "file_count": len(pdf_files),
                "size": pdf_size
            },
            "metadata": {
                "pdf_count": metadata_count,
                "index_count": len(pdf_files),  # 실제 PDF 파일 수를 기준으로
                "hash_count": len(pdf_files)    # 실제 PDF 파일 수를 기준으로
            }
        }
        return status
    except Exception as e:
        logger.error(f"데이터베이스 상태 확인 중 오류 발생: {str(e)}")
        return {"error": str(e)}

def verify_data_persistence() -> Dict:
    """데이터 지속성 상태를 검증합니다."""
    try:
        status = {
            "chroma_db": {
                "status": "ok" if os.path.exists(CHROMA_DB_PATH) else "missing",
                "collections": list_available_collections()
            },
            "pdf_storage": {
                "status": "ok" if os.path.exists(PDF_STORAGE_PATH) else "missing",
                "files": os.listdir(PDF_STORAGE_PATH) if os.path.exists(PDF_STORAGE_PATH) else []
            },
            "metadata": {
                "status": "ok" if all(os.path.exists(f) for f in [PDF_METADATA_PATH, PDF_INDEX_PATH, PDF_HASH_PATH]) else "incomplete",
                "files": {
                    "metadata": os.path.exists(PDF_METADATA_PATH),
                    "index": os.path.exists(PDF_INDEX_PATH),
                    "hashes": os.path.exists(PDF_HASH_PATH)
                }
            }
        }
        return status
    except Exception as e:
        logger.error(f"데이터 지속성 검증 중 오류 발생: {str(e)}")
        return {"error": str(e)}

def recover_data_if_needed() -> bool:
    """필요한 경우 데이터를 복구합니다."""
    try:
        # 디렉토리 존재 확인 및 생성
        os.makedirs(CHROMA_DB_PATH, exist_ok=True)
        os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
        
        # ChromaDB 초기화 확인
        if not any(col == "rag_collection" for col in list_available_collections()):
            logger.info("ChromaDB 컬렉션 재생성")
            global vectorstore
            vectorstore = Chroma(
                embedding_function=embeddings,
                persist_directory=CHROMA_DB_PATH,
                collection_name="rag_collection"
            )
        
        # 메타데이터 파일 확인 및 복구
        if not os.path.exists(PDF_METADATA_PATH):
            save_pdf_metadata()
        if not os.path.exists(PDF_INDEX_PATH):
            save_pdf_index()
        if not os.path.exists(PDF_HASH_PATH):
            save_pdf_hashes()
        
        # PDF 파일과 메타데이터 동기화
        stored_pdfs = set(os.listdir(PDF_STORAGE_PATH))
        metadata_pdfs = set(f"{data['id']}_{data['filename']}" 
                          for data in pdf_metadata.values())
        
        # 메타데이터에 있지만 파일이 없는 경우 처리
        for pdf_id, data in pdf_metadata.items():
            expected_file = f"{pdf_id}_{data['filename']}"
            if expected_file not in stored_pdfs:
                logger.warning(f"PDF 파일 누락: {expected_file}")
                data["status"] = "missing"
        
        # 파일은 있지만 메타데이터가 없는 경우 처리
        for pdf_file in stored_pdfs:
            if pdf_file not in metadata_pdfs:
                logger.warning(f"메타데이터 누락: {pdf_file}")
                # 파일 삭제 또는 메타데이터 생성 로직 추가 가능
        
        save_pdf_metadata()
        return True
        
    except Exception as e:
        logger.error(f"데이터 복구 중 오류 발생: {str(e)}")
        return False

# 프로그램 시작 시 데이터 검증 및 복구
def initialize_data():
    """프로그램 시작 시 데이터를 초기화하고 검증합니다."""
    try:
        # 기존 데이터 로드
        load_pdf_metadata()
        load_pdf_index()
        load_pdf_hashes()
        
        # 데이터 지속성 검증
        status = verify_data_persistence()
        if any(v.get("status") != "ok" for v in status.values() if isinstance(v, dict) and "status" in v):
            logger.warning("데이터 지속성 문제 발견, 복구 시도")
            if not recover_data_if_needed():
                logger.error("데이터 복구 실패")
                return False
        
        # ChromaDB 초기화
        global vectorstore
        vectorstore = initialize_chroma()
        if vectorstore is None:
            logger.error("ChromaDB 초기화 실패")
            return False
        
        logger.info("데이터 초기화 완료")
        return True
        
    except Exception as e:
        logger.error(f"데이터 초기화 중 오류 발생: {str(e)}")
        return False

# 대신 main.py에서 명시적으로 호출하도록 변경
def get_initialized_vectorstore():
    """초기화된 vectorstore를 반환합니다."""
    global vectorstore
    if vectorstore is None:
        initialize_data()
    return vectorstore

def process_directory_pdfs(directory_path: str) -> Dict[str, bool]:
    """
    지정된 디렉토리 내의 모든 PDF 파일을 처리합니다.
    """
    results = {}
    try:
        # 디렉토리 내의 모든 PDF 파일 찾기
        pdf_files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"디렉토리에 PDF 파일이 없습니다: {directory_path}")
            return results
        
        logger.info(f"처리할 PDF 파일 수: {len(pdf_files)}")
        
        # 각 PDF 파일 처리
        for pdf_file in pdf_files:
            pdf_path = os.path.join(directory_path, pdf_file)
            logger.info(f"PDF 처리 시작: {pdf_file}")
            
            try:
                success = process_and_embed_pdf(pdf_path)
                results[pdf_file] = success
                
                if success:
                    logger.info(f"PDF 처리 성공: {pdf_file}")
                else:
                    logger.error(f"PDF 처리 실패: {pdf_file}")
                    
            except Exception as e:
                logger.error(f"PDF 처리 중 오류 발생: {pdf_file} - {str(e)}")
                results[pdf_file] = False
        
        # 처리 결과 요약
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"PDF 처리 완료 - 성공: {success_count}, 실패: {len(results) - success_count}")
        
        return results
        
    except Exception as e:
        logger.error(f"디렉토리 처리 중 오류 발생: {str(e)}")
        return results

def process_multiple_pdfs(pdf_files: List[Tuple[str, bytes]]) -> Dict[str, bool]:
    """
    여러 PDF 파일을 처리합니다.
    
    Args:
        pdf_files: (파일명, 파일내용) 튜플의 리스트
    
    Returns:
        Dict[str, bool]: 파일명을 키로 하고 처리 성공 여부를 값으로 하는 딕셔너리
    """
    results = {}
    for pdf_name, pdf_content in pdf_files:
        try:
            # 임시 파일 생성
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_content)
                temp_file_path = tmp_file.name

            try:
                # PDF 처리
                success = process_and_embed_pdf(temp_file_path)
                results[pdf_name] = success
                
                if success:
                    logger.info(f"PDF 처리 성공: {pdf_name}")
                else:
                    logger.error(f"PDF 처리 실패: {pdf_name}")
                    
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    
        except Exception as e:
            logger.error(f"PDF 처리 중 오류 발생: {pdf_name} - {str(e)}")
            results[pdf_name] = False
    
    # 처리 결과 요약
    success_count = sum(1 for success in results.values() if success)
    logger.info(f"PDF 처리 완료 - 성공: {success_count}, 실패: {len(results) - success_count}")
    
    return results