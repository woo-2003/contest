import sys
import os
# 현재 파일(app.py)의 절대 경로를 기준으로 상위 디렉토리(src)를 sys.path에 추가
# __file__ 은 현재 실행 중인 스크립트의 경로입니다.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multi_agent_chatbot.agent_logic import run_graph
from multi_agent_chatbot.rag_handler import process_and_embed_pdf, PDF_STORAGE_PATH

import gradio as gr
from PIL import Image
from typing import List, Tuple, Optional

# from .agent_logic import run_graph
# from .rag_handler import process_and_embed_pdf, PDF_STORAGE_PATH

# --- Gradio 인터페이스 ---
def chat_interface(message: str, history: List[Tuple[str, str]], image_upload: Optional[Image.Image]):
    """Gradio 챗봇 인터페이스 함수"""
    print(f"User query: {message}")
    if image_upload:
        print(f"Image uploaded: {type(image_upload)}")
    
    # run_graph 함수는 PIL Image 객체를 직접 받도록 수정됨
    response_text = run_graph(message, history, image_upload)
    
    history.append((message, response_text))
    return "", history, None # 입력창 비우기, 업데이트된 히스토리, 이미지 업로드 초기화

def process_pdf_upload(pdf_file):
    """Gradio PDF 업로드 처리 함수"""
    if pdf_file is not None:
        # Gradio File 컴포넌트는 임시 파일 경로를 반환
        file_path = pdf_file.name 
        
        # (선택적) 업로드된 파일을 영구적인 위치로 복사/이동
        # filename = os.path.basename(file_path)
        # permanent_file_path = os.path.join(PDF_STORAGE_PATH, filename)
        # import shutil
        # shutil.copy(file_path, permanent_file_path)
        # print(f"Copied PDF to {permanent_file_path}")
        
        success = process_and_embed_pdf(file_path) # 임시 파일 경로 직접 사용
        if success:
            return f"'{os.path.basename(file_path)}' 파일이 성공적으로 처리되어 RAG DB에 추가되었습니다."
        else:
            return f"'{os.path.basename(file_path)}' 파일 처리 중 오류가 발생했습니다."
    return "PDF 파일이 업로드되지 않았습니다."

# Gradio 앱 구성
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 멀티 에이전트 AI 챗봇 (Ollama & RAG)")
    gr.Markdown("코딩/수학 문제는 `deepseek-coder`, 복잡한 추론은 `llama3`, 일반 질문은 `gemma`가 처리합니다. PDF를 업로드하여 RAG 기능을 사용할 수 있고, 이미지도 업로드하여 분석할 수 있습니다.")

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="대화 내용", height=600, bubble_full_width=False)
            
            with gr.Row():
                image_input = gr.Image(type="pil", label="이미지 업로드 (선택)", sources=["upload"], height=150, width=150)
                user_input = gr.Textbox(label="질문 입력", placeholder="여기에 질문을 입력하세요...", scale=3)
            
            submit_button = gr.Button("전송", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("## RAG 설정")
            pdf_upload = gr.File(label="PDF 파일 업로드 (RAG 학습용)", file_types=[".pdf"])
            pdf_status = gr.Textbox(label="PDF 처리 상태", interactive=False)
            pdf_upload.upload(process_pdf_upload, inputs=pdf_upload, outputs=pdf_status)
            
            gr.Markdown("## 모델 정보")
            gr.Markdown(
                """
                - **일반 질문**: `qwen3:latest`
                - **이미지/PDF**: `llava:7b`
                - **복잡한 추론/이미지**: `llama3.2:latest`
                - **코딩/수학**: `deepseek-r1:latest`
                - **임베딩**: `nomic-embed-text:latest`
                - **토큰 컨텍스트**: 4096
                """
            )
            gr.Markdown("---")
            gr.Markdown("### 사용 팁:\n"
                        "- PDF를 업로드하면 해당 내용 기반으로 답변합니다.\n"
                        "- 이미지와 함께 질문하면 이미지를 분석하여 답변에 활용합니다.\n"
                        "- '코드 짜줘', '수학 문제 풀어줘' 등으로 특정 에이전트를 유도할 수 있습니다.")


    # 이벤트 핸들러 연결
    #Textbox에서 엔터키 또는 버튼 클릭시
    user_input.submit(chat_interface, inputs=[user_input, chatbot, image_input], outputs=[user_input, chatbot, image_input])
    submit_button.click(chat_interface, inputs=[user_input, chatbot, image_input], outputs=[user_input, chatbot, image_input])

def main():
    demo.launch(server_name="127.0.0.1", share = True) # Docker 등에서 실행 시 외부 접속 허용

if __name__ == "__main__":
    main()