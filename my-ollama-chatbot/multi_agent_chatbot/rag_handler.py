import os
from pathlib import Path
from typing import List, Tuple, Optional

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from .llm_config import embeddings, llm_coding # JS 변환에 코딩 LLM 사용
from .utils import extract_javascript_from_text, convert_js_to_python_code

CHROMA_DB_PATH = str(Path(__file__).parent.parent.parent / "data" / "chroma_db")
PDF_STORAGE_PATH = str(Path(__file__).parent.parent.parent / "data" / "pdfs")

os.makedirs(CHROMA_DB_PATH, exist_ok=True)
os.makedirs(PDF_STORAGE_PATH, exist_ok=True)

# ChromaDB 클라이언트 초기화 (메모리 또는 영구 저장소)
vectorstore = Chroma(
    embedding_function=embeddings,
    persist_directory=CHROMA_DB_PATH,
    collection_name="rag_collection"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

def process_and_embed_pdf(pdf_file_path: str) -> bool:
    """
    PDF 파일을 처리하고, JavaScript를 Python으로 변환 후 Vector DB에 저장합니다.
    성공 시 True, 실패 시 False 반환.
    """
    try:
        print(f"Processing PDF: {pdf_file_path}")
        # loader = PyPDFLoader(pdf_file_path) # 간단한 텍스트 추출
        loader = UnstructuredPDFLoader(pdf_file_path, mode="elements") # 이미지, 테이블 등 고려
        
        docs = loader.load()
        
        processed_docs: List[Document] = []
        for doc in docs:
            content = doc.page_content
            js_codes = extract_javascript_from_text(content)
            
            if js_codes:
                print(f"Found {len(js_codes)} JavaScript blocks in a chunk. Converting to Python...")
                for js_code in js_codes:
                    python_code = convert_js_to_python_code(js_code, llm_coding)
                    # 원본 JS 코드를 변환된 Python 코드로 대체
                    # 주의: 이 방식은 매우 단순하며, 복잡한 문서 구조에서는 부정확할 수 있음
                    content = content.replace(js_code, f"\n'''\nOriginal JavaScript:\n{js_code}\n'''\n\n'''\nConverted Python:\n{python_code}\n'''\n")
                
                # 변환된 내용을 포함하는 새 Document 객체 생성 또는 기존 Document 업데이트
                # 여기서는 간단히 page_content를 업데이트합니다.
                doc.page_content = content

            processed_docs.append(doc)

        if not processed_docs:
            print(f"No text could be extracted from {pdf_file_path}")
            return False

        split_docs = text_splitter.split_documents(processed_docs)
        
        if not split_docs:
            print(f"No text chunks generated after splitting for {pdf_file_path}")
            return False
            
        vectorstore.add_documents(split_docs)
        vectorstore.persist() # 변경사항 디스크에 저장
        print(f"Successfully processed and embedded {pdf_file_path}")
        return True
    except Exception as e:
        print(f"Error processing PDF {pdf_file_path}: {e}")
        import traceback
        traceback.print_exc()
        return False

def query_rag(query: str, k: int = 5) -> List[Document]:
    """Vector DB에서 관련 문서를 검색합니다."""
    try:
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        relevant_docs = retriever.invoke(query)
        return relevant_docs
    except Exception as e:
        print(f"Error querying RAG: {e}")
        return []

def get_rag_retriever(k: int = 5):
    return vectorstore.as_retriever(search_kwargs={"k": k})

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