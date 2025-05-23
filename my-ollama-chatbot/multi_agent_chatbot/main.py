import sys
import os
from PIL import Image
from typing import List, Tuple, Optional
import streamlit as st
import tempfile
import time

# 현재 파일의 절대 경로를 기준으로 상위 디렉토리를 sys.path에 추가
current_file_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_file_dir)
sys.path.insert(0, parent_dir)

from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, PDF_STORAGE_PATH

# 페이지 설정 (반드시 첫 번째 Streamlit 명령어여야 함)
st.set_page_config(
    page_title="멀티 에이전트 AI 챗봇",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일 정의
st.markdown("""
<style>
    /* 전체 페이지 스타일 */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* 채팅 메시지 스타일 */
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: row;
        align-items: flex-start;
        gap: 1rem;
    }
    
    .chat-message.user {
        background-color: #f0f2f6;
    }
    
    .chat-message.assistant {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
    }
    
    .chat-message .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        object-fit: cover;
    }
    
    .chat-message .message {
        flex: 1;
        padding: 0.5rem 0;
    }
    
    /* 입력 영역 스타일 */
    .stTextInput > div > div > input {
        border-radius: 1rem;
        padding: 0.75rem 1rem;
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 1rem;
        padding: 0.5rem 1.5rem;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        padding: 2rem 1rem;
        background-color: #f8f9fa;
    }

    /* 사이드바 헤더 스타일 */
    .sidebar-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e0e0e0;
    }

    .sidebar-header img {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .sidebar-header h1 {
        font-size: 1.5rem;
        color: #1f1f1f;
        margin: 0;
        font-weight: 600;
    }

    /* 사이드바 섹션 스타일 */
    .sidebar-section {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    .sidebar-section h2 {
        font-size: 1.1rem;
        color: #1f1f1f;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* 파일 업로더 스타일 */
    .stFileUploader > div {
        border-radius: 0.5rem;
        border: 2px dashed #e0e0e0;
        background: white;
        padding: 1rem;
    }

    /* 모델 정보 스타일 */
    .model-info {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.6;
    }

    .model-info strong {
        color: #1f1f1f;
    }

    /* 사용 팁 스타일 */
    .usage-tips {
        font-size: 0.9rem;
        color: #666;
        line-height: 1.6;
    }

    .usage-tips li {
        margin-bottom: 0.5rem;
    }

    /* 메인 컨테이너 스타일 */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        position: relative;
    }

    /* 채팅 컨테이너 스타일 */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 1rem;
        margin-bottom: 200px; /* 입력 영역 높이만큼 여백 추가 */
    }

    /* 입력 컨테이너 스타일 */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 1000;
        max-width: 1200px;
        margin: 0 auto;
        display: flex;
        flex-direction: column;
        gap: 1rem;
    }

    /* 이미지 업로더 컨테이너 스타일 */
    .image-uploader-container {
        margin-bottom: 1rem;
    }

    /* 채팅 입력창 스타일 */
    .chat-input-container {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming" not in st.session_state:
    st.session_state.streaming = False

def process_pdf_upload(pdf_file):
    """PDF 파일 업로드 처리 함수"""
    if pdf_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(pdf_file.getvalue())
            temp_file_path = tmp_file.name

        try:
            success = process_and_embed_pdf(temp_file_path)
            if success:
                return f"'{pdf_file.name}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다."
            else:
                return f"'{pdf_file.name}' 파일 처리 중 오류가 발생했습니다."
        finally:
            os.unlink(temp_file_path)
    return "PDF 파일이 업로드되지 않았습니다."

def stream_response(response_text):
    """응답을 스트리밍하는 함수"""
    response_container = st.empty()
    full_response = ""
    
    for chunk in response_text.split():
        full_response += chunk + " "
        response_container.markdown(full_response + "▌")
        time.sleep(0.05)  # 스트리밍 효과를 위한 지연
    
    response_container.markdown(full_response)
    return full_response

def main():
    # 사이드바 설정
    with st.sidebar:
        # 사이드바 헤더
        st.markdown("""
        <div class="sidebar-header">
            <img src="https://via.placeholder.com/150" alt="Logo">
            <h1>멀티 에이전트 AI 챗봇</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # RAG 설정
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h2>📚 RAG 설정</h2>', unsafe_allow_html=True)
        pdf_file = st.file_uploader("PDF 파일 업로드", type=['pdf'])
        if pdf_file:
            with st.spinner("PDF 처리 중..."):
                status = process_pdf_upload(pdf_file)
                st.info(status)
        st.markdown('</div>', unsafe_allow_html=True)

        # 모델 정보
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h2>🤖 모델 정보</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="model-info">
            <p><strong>코딩/수학</strong>: deepseek-coder:6.7b</p>
            <p><strong>복잡한 추론/이미지</strong>: llama3:8b</p>
            <p><strong>일반 질문</strong>: gemma:2b</p>
            <p><strong>임베딩</strong>: nomic-embed-text</p>
            <p><strong>벡터DB</strong>: ChromaDB</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 사용 팁
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h2>💡 사용 팁</h2>', unsafe_allow_html=True)
        st.markdown("""
        <div class="usage-tips">
            <ul>
                <li>PDF를 업로드하면 해당 내용 기반으로 답변합니다.</li>
                <li>이미지와 함께 질문하면 이미지를 분석하여 답변에 활용합니다.</li>
                <li>'코드 짜줘', '수학 문제 풀어줘' 등으로 특정 에이전트를 유도할 수 있습니다.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 메인 컨테이너
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # 채팅 메시지 표시 영역 (상단)
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <img class="avatar" src="https://via.placeholder.com/40" alt="User">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
            if "image" in message:
                st.image(message["image"], width=300)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <img class="avatar" src="https://via.placeholder.com/40" alt="Assistant">
                <div class="message">{message["content"]}</div>
            </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # 입력 영역 (하단)
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    
    # 이미지 업로드
    st.markdown('<div class="image-uploader-container">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])
    image = None
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    # 사용자 입력
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    if prompt := st.chat_input("여기에 질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        if image:
            st.session_state.messages[-1]["image"] = image

        # 챗봇 응답 생성
        with st.spinner("생각 중..."):
            try:
                response = run_graph(
                    prompt,
                    [(m["content"], "") for m in st.session_state.messages if m["role"] == "user"],
                    image
                )
                # 스트리밍 응답
                full_response = stream_response(response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 