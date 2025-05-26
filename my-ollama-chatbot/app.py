import gradio as gr
import os
from pathlib import Path
import tempfile
import shutil
from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, get_processed_pdfs
from PIL import Image
import json

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

# 현재 테마 저장
current_theme = "kakao"

def apply_theme(theme_name):
    """테마를 적용하고 CSS를 반환합니다."""
    global current_theme
    current_theme = theme_name
    theme = THEMES[theme_name]
    
    return f"""
    .user-message {{
        background-color: {theme['user_bubble']} !important;
        color: {theme['user_text']} !important;
        border-radius: 15px 15px 0 15px !important;
        padding: 10px 15px !important;
        margin: 5px 0 !important;
        max-width: 80% !important;
        margin-left: auto !important;
    }}
    
    .bot-message {{
        background-color: {theme['bot_bubble']} !important;
        color: {theme['bot_text']} !important;
        border-radius: 15px 15px 15px 0 !important;
        padding: 10px 15px !important;
        margin: 5px 0 !important;
        max-width: 80% !important;
        margin-right: auto !important;
    }}
    
    .chat-container {{
        background-color: {theme['background']} !important;
        padding: 20px !important;
        border-radius: 10px !important;
    }}
    """

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

def process_message(message, history, image):
    """메시지를 처리하고 응답을 생성합니다."""
    if not message and not image:
        return history, "메시지나 이미지를 입력해주세요."
    
    # 이미지가 있는 경우 PIL Image로 변환
    image_pil = None
    if image is not None:
        image_pil = Image.open(image)
    
    # 에이전트 실행
    response = run_graph(message, history, image_pil)
    
    # 히스토리 업데이트
    history.append((message, response))
    
    return history, ""

def create_ui():
    """UI를 생성합니다."""
    with gr.Blocks(css=apply_theme(current_theme)) as demo:
        gr.Markdown("# 멀티 에이전트 챗봇")
        
        with gr.Row():
            with gr.Column(scale=3):
                # 채팅 인터페이스
                chatbot = gr.Chatbot(
                    label="채팅",
                    elem_classes=["chat-container"],
                    bubble_full_width=False
                )
                
                with gr.Row():
                    msg = gr.Textbox(
                        label="메시지",
                        placeholder="메시지를 입력하세요...",
                        show_label=False,
                        container=False
                    )
                    image_input = gr.Image(
                        label="이미지",
                        type="filepath",
                        show_label=False
                    )
                
                with gr.Row():
                    submit_btn = gr.Button("전송")
                    clear_btn = gr.Button("대화 초기화")
            
            with gr.Column(scale=1):
                # 테마 선택
                gr.Markdown("### 채팅 테마")
                theme_radio = gr.Radio(
                    choices=["카카오톡", "인스타그램", "라인", "Copilot"],
                    value="카카오톡",
                    label="테마 선택"
                )
                
                # PDF 업로드 섹션
                gr.Markdown("### PDF 파일 업로드")
                pdf_upload = gr.File(
                    label="PDF 파일 선택",
                    file_types=[".pdf"]
                )
                pdf_status = gr.Textbox(
                    label="PDF 처리 상태",
                    interactive=False
                )
                pdf_list = gr.Textbox(
                    label="처리된 PDF 목록",
                    interactive=False,
                    lines=5
                )
                refresh_pdf_btn = gr.Button("PDF 목록 새로고침")
        
        # 이벤트 핸들러
        theme_radio.change(
            fn=lambda x: apply_theme(x.lower()),
            inputs=[theme_radio],
            outputs=[demo]
        )
        
        submit_btn.click(
            process_message,
            inputs=[msg, chatbot, image_input],
            outputs=[chatbot, msg]
        ).then(
            lambda: None,
            None,
            image_input,
            _js="() => document.querySelector('input[type=file]').value = ''"
        )
        
        msg.submit(
            process_message,
            inputs=[msg, chatbot, image_input],
            outputs=[chatbot, msg]
        ).then(
            lambda: None,
            None,
            image_input,
            _js="() => document.querySelector('input[type=file]').value = ''"
        )
        
        clear_btn.click(lambda: None, None, chatbot, queue=False)
        
        pdf_upload.change(
            process_pdf_upload,
            inputs=[pdf_upload],
            outputs=[pdf_status]
        )
        
        refresh_pdf_btn.click(
            get_pdf_list,
            inputs=[],
            outputs=[pdf_list]
        )
        
        # 초기 PDF 목록 로드
        demo.load(
            get_pdf_list,
            inputs=[],
            outputs=[pdf_list]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_ui()
    demo.launch(share=True) 