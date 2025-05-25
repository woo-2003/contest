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
        background-color: #fafafa;
        padding: 0;
    }
    
    /* 채팅 메시지 스타일 */
    .chat-message {
        display: flex;
        align-items: flex-start;
        margin: 8px 0;
        max-width: 60%;
        position: relative;
        width: 100%;
    }
    
    /* 사용자 메시지 스타일 */
    .chat-message.user {
        margin-left: auto;
        flex-direction: row;
        justify-content: flex-end;
        padding-right: 0;
        width: 100%;
    }
    
    .chat-message.user .message {
        background-color: #0095f6;
        color: white;
        border-radius: 22px;
        border-bottom-right-radius: 4px;
        padding: 12px 20px;
        margin-right: 12px;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        order: 1;
        max-width: calc(100% - 48px);
    }
    
    .chat-message.user .avatar {
        order: 2;
        flex-shrink: 0;
    }
    
    /* AI 메시지 스타일 */
    .chat-message.assistant {
        margin-right: auto;
        flex-direction: row;
        justify-content: flex-start;
        padding-left: 0;
        width: 100%;
    }
    
    .chat-message.assistant .message {
        background-color: #efefef;
        color: #262626;
        border-radius: 22px;
        border-bottom-left-radius: 4px;
        padding: 12px 20px;
        margin-left: 12px;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
        order: 2;
        max-width: calc(100% - 48px);
    }
    
    .chat-message.assistant .avatar {
        order: 1;
        flex-shrink: 0;
    }
    
    /* 아바타 스타일 */
    .chat-message .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        background-color: white;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* 메시지 내 이미지 스타일 */
    .chat-message img {
        max-width: 300px;
        border-radius: 12px;
        margin-top: 8px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    /* 코드 블록 스타일 */
    .chat-message pre {
        background-color: rgba(0, 0, 0, 0.05);
        padding: 16px;
        border-radius: 8px;
        overflow-x: auto;
        margin: 8px 0;
        font-size: 14px;
    }
    
    .chat-message code {
        font-family: 'Consolas', monospace;
        font-size: 14px;
    }
    
    /* 입력 영역 스타일 */
    .stTextInput > div > div > input {
        border-radius: 24px;
        padding: 14px 24px;
        border: 1px solid #dbdbdb;
        background-color: white;
        font-size: 15px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }
    
    /* 버튼 스타일 */
    .stButton > button {
        border-radius: 24px;
        padding: 10px 20px;
        background-color: #0095f6;
        color: white;
        border: none;
        transition: background-color 0.2s ease;
        font-size: 15px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background-color: #0081d6;
    }
    
    /* 사이드바 스타일 */
    .css-1d391kg {
        padding: 2rem 1.5rem;
        background-color: white;
        border-right: 1px solid #dbdbdb;
        width: 300px !important;
    }

    /* 사이드바 헤더 스타일 */
    .sidebar-header {
        text-align: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #dbdbdb;
    }

    .sidebar-header img {
        width: 100px;
        height: 100px;
        border-radius: 50%;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .sidebar-header h1 {
        font-size: 1.8rem;
        color: #262626;
        margin: 0;
        font-weight: 600;
    }

    /* 사이드바 섹션 스타일 */
    .sidebar-section {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .sidebar-section h2 {
        font-size: 1.2rem;
        color: #262626;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }

    /* 파일 업로더 스타일 */
    .stFileUploader > div {
        border-radius: 12px;
        border: 2px dashed #dbdbdb;
        background: white;
        padding: 1.5rem;
    }

    /* 모델 정보 스타일 */
    .model-info {
        font-size: 1rem;
        color: #8e8e8e;
        line-height: 1.6;
    }

    .model-info strong {
        color: #262626;
    }

    /* 사용 팁 스타일 */
    .usage-tips {
        font-size: 1rem;
        color: #8e8e8e;
        line-height: 1.6;
    }

    .usage-tips li {
        margin-bottom: 0.8rem;
    }

    /* 메인 컨테이너 스타일 */
    .main-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        max-width: 1200px;
        margin: 0 auto;
        position: relative;
        background-color: #fafafa;
    }

    /* 채팅 컨테이너 스타일 */
    .chat-container {
        flex: 1;
        overflow-y: auto;
        padding: 2rem;
        margin-bottom: 200px;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        width: 100%;
        max-width: 1200px;
        margin-left: auto;
        margin-right: auto;
    }

    /* 입력 컨테이너 스타일 */
    .bottom-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        border-top: 1px solid #dbdbdb;
        z-index: 1000;
        box-shadow: 0 -1px 2px rgba(0, 0, 0, 0.05);
    }

    .bottom-content {
        max-width: 1200px;
        margin: 0 auto;
        padding: 1.5rem;
    }

    /* 입력 컨테이너 스타일 */
    .chat-input {
        margin-bottom: 1.5rem;
    }

    /* 이미지 업로더 컨테이너 스타일 */
    .image-uploader {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    /* 메시지 컨테이너 스타일 */
    .element-container {
        width: 100% !important;
        max-width: 1200px !important;
        margin-left: auto !important;
        margin-right: auto !important;
    }

    /* 반응형 스타일 */
    @media (max-width: 1200px) {
        .stApp {
            max-width: 100%;
            padding: 0 1rem;
        }
        
        .main-container {
            max-width: 100%;
        }
        
        .bottom-content {
            max-width: 100%;
        }
        
        .chat-container {
            padding: 1.5rem;
        }
        
        .chat-message {
            max-width: 70%;
        }
    }

    @media (max-width: 768px) {
        .chat-container {
            padding: 1rem;
        }
        
        .chat-message {
            max-width: 85%;
        }
        
        .bottom-content {
            padding: 1rem;
        }
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
        except Exception as e:
            return f"'{pdf_file.name}' 파일 처리 중 오류가 발생했습니다: {str(e)}"
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
    
    # 메시지 표시를 위한 컨테이너
    messages_container = st.container()
    
    # 메시지 표시
    with messages_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user">
                    <div class="message">{message["content"]}</div>
                    <div class="avatar">👤</div>
                </div>
                """, unsafe_allow_html=True)
                if "image" in message:
                    st.image(message["image"], width=300)
            else:
                st.markdown(f"""
                <div class="chat-message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message">{message["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # 하단 고정 영역
    st.markdown('<div class="bottom-container">', unsafe_allow_html=True)
    st.markdown('<div class="bottom-content">', unsafe_allow_html=True)

    # 이미지 업로드
    st.markdown('<div class="image-uploader">', unsafe_allow_html=True)
    uploaded_image = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])
    image = None
    if uploaded_image:
        image = Image.open(uploaded_image)
        st.image(image, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    # 사용자 입력
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    if prompt := st.chat_input("여기에 질문을 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 메시지 표시 업데이트
        with messages_container:
            st.markdown(f"""
            <div class="chat-message user">
                <div class="message">{prompt}</div>
                <div class="avatar">👤</div>
            </div>
            """, unsafe_allow_html=True)
            if image:
                st.session_state.messages[-1]["image"] = image
                st.image(image, width=300)

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
                
                # AI 응답 표시
                with messages_container:
                    st.markdown(f"""
                    <div class="chat-message assistant">
                        <div class="avatar">🤖</div>
                        <div class="message">{full_response}</div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"오류가 발생했습니다: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # bottom-content 닫기
    st.markdown('</div>', unsafe_allow_html=True)  # bottom-container 닫기
    st.markdown('</div>', unsafe_allow_html=True)  # main-container 닫기

if __name__ == "__main__":
    main()