import streamlit as st
import os
from pathlib import Path
import tempfile
import shutil
from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, get_processed_pdfs
from PIL import Image
import json

# 페이지 설정
st.set_page_config(
    page_title="멀티 에이전트 챗봇",
    page_icon="🤖",
    layout="wide"
)

# 테마 설정
THEMES = {
    "kakao": {
        "user_bubble": "#FEE500",  # 카카오 노란색
        "bot_bubble": "#FFFFFF",   # 흰색
        "user_text": "#000000",    # 검정색
        "bot_text": "#000000",     # 검정색
        "background": "#F5F5F5"    # 연한 회색
    },
    "instagram": {
        "user_bubble": "#0095F6",  # 인스타그램 파란색
        "bot_bubble": "#EFEFEF",   # 연한 회색
        "user_text": "#FFFFFF",    # 흰색
        "bot_text": "#000000",     # 검정색
        "background": "#FAFAFA"    # 매우 연한 회색
    },
    "line": {
        "user_bubble": "#00B900",  # 라인 초록색
        "bot_bubble": "#FFFFFF",   # 흰색
        "user_text": "#FFFFFF",    # 흰색
        "bot_text": "#000000",     # 검정색
        "background": "#F5F5F5"    # 연한 회색
    },
    "copilot": {
        "user_bubble": "#0078D4",  # MS 파란색
        "bot_bubble": "#F3F2F1",   # 연한 회색
        "user_text": "#FFFFFF",    # 흰색
        "bot_text": "#000000",     # 검정색
        "background": "#FFFFFF"    # 흰색
    }
}

def apply_theme(theme_name):
    """테마를 적용합니다."""
    theme = THEMES[theme_name]
    
    # CSS 스타일 적용
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {theme['background']};
        }}
        .user-message {{
            background-color: {theme['user_bubble']};
            color: {theme['user_text']};
            padding: 10px 15px;
            border-radius: 15px 15px 0 15px;
            margin: 5px 0;
            max-width: 80%;
            margin-left: auto;
        }}
        .bot-message {{
            background-color: {theme['bot_bubble']};
            color: {theme['bot_text']};
            padding: 10px 15px;
            border-radius: 15px 15px 15px 0;
            margin: 5px 0;
            max-width: 80%;
            margin-right: auto;
        }}
        </style>
    """, unsafe_allow_html=True)

def process_pdf_upload(file):
    """PDF 파일을 처리하고 임베딩합니다."""
    if file is None:
        return "PDF 파일을 선택해주세요."
    
    try:
        # 임시 파일 생성
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            shutil.copy2(file.name, tmp_file.name)
            tmp_path = tmp_file.name
        
        # PDF 처리
        success = process_and_embed_pdf(tmp_path)
        
        # 임시 파일 삭제
        os.unlink(tmp_path)
        
        if success:
            return f"PDF 파일 '{os.path.basename(file.name)}'이(가) 성공적으로 처리되었습니다."
        else:
            return f"PDF 파일 '{os.path.basename(file.name)}' 처리 중 오류가 발생했습니다."
    except Exception as e:
        return f"PDF 처리 중 오류 발생: {str(e)}"

def get_pdf_list():
    """처리된 PDF 파일 목록을 반환합니다."""
    pdfs = get_processed_pdfs()
    if not pdfs:
        return "처리된 PDF 파일이 없습니다."
    
    return "\n".join([f"- {pdf['filename']} (상태: {pdf['status']})" for pdf in pdfs])

def main():
    st.title("멀티 에이전트 챗봇")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("설정")
        
        # 테마 선택
        st.markdown("### 🎨 채팅 테마")
        theme = st.radio(
            "원하는 테마를 선택하세요",
            ["카카오톡", "인스타그램", "라인", "Copilot"],
            index=0,
            label_visibility="visible"
        )
        
        # 테마 적용
        apply_theme(theme.lower())
        
        st.markdown("---")  # 구분선 추가
        
        # PDF 업로드
        st.markdown("### 📄 PDF 파일 업로드")
        uploaded_file = st.file_uploader(
            "PDF 파일을 선택하세요",
            type=['pdf'],
            help="분석할 PDF 파일을 업로드하세요"
        )
        
        if uploaded_file is not None:
            status = process_pdf_upload(uploaded_file)
            st.info(status)
        
        # PDF 목록
        st.markdown("### 📚 처리된 PDF 목록")
        pdf_list = get_pdf_list()
        st.text_area("", pdf_list, height=150)
        
        if st.button("🔄 PDF 목록 새로고침", use_container_width=True):
            st.experimental_rerun()
    
    # 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 채팅 히스토리 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # 메시지 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 봇 응답 생성
        with st.chat_message("assistant"):
            response = run_graph(prompt, st.session_state.messages, None)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    # 이미지 업로드
    uploaded_image = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="업로드된 이미지")
        
        # 이미지 분석 요청
        if st.button("이미지 분석"):
            with st.chat_message("assistant"):
                response = run_graph("이 이미지를 분석해줘", st.session_state.messages, image)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main() 