from PIL import Image
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage

from .llm_config import llm_image_analysis # 또는 llm_reasoning (llama3가 멀티모달 지원 시)
from .utils import pil_to_base64

def analyze_image_with_llm(image: Image.Image, prompt: str) -> str:
    """
    PIL Image 객체와 프롬프트를 받아 LLM으로 이미지를 분석하고 텍스트 설명을 반환합니다.
    """
    try:
        base64_image = pil_to_base64(image)
        
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ]
            )
        ]
        
        # llm_image_analysis 또는 llm_reasoning 모델 사용
        # 모델이 멀티모달 입력을 지원해야 함 (예: llava, gpt-4-vision-preview)
        # Ollama에서 llama3가 멀티모달을 지원하는지 확인 필요. 지원하지 않으면 llava 같은 모델 사용.
        response = llm_image_analysis.invoke(messages)
        
        return response.content if hasattr(response, 'content') else str(response)

    except Exception as e:
        print(f"Error analyzing image: {e}")
        return f"이미지 분석 중 오류 발생: {e}"