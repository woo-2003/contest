from PIL import Image
from typing import Optional, List
from langchain_core.messages import HumanMessage, SystemMessage
import base64
import io
import logging
import pytesseract
from PIL import ImageEnhance, ImageFilter
import hashlib
from functools import lru_cache

from .llm_config import llm_image

# 로깅 설정
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 이미지 캐시
image_cache = {}

@lru_cache(maxsize=100)
def get_image_hash(image_bytes: bytes) -> str:
    """이미지의 해시값을 생성합니다."""
    return hashlib.md5(image_bytes).hexdigest()

def convert_image_format(image: Image.Image) -> Image.Image:
    """
    이미지 형식을 변환하여 처리 가능한 상태로 만듭니다.
    """
    try:
        # RGBA 이미지를 RGB로 변환
        if image.mode in ('RGBA', 'LA'):
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            return background
        # 그레이스케일 이미지를 RGB로 변환
        elif image.mode in ('L', '1'):
            return image.convert('RGB')
        # 이미 RGB인 경우 그대로 반환
        elif image.mode == 'RGB':
            return image
        # 기타 형식은 RGB로 변환
        else:
            return image.convert('RGB')
    except Exception as e:
        logger.error(f"이미지 형식 변환 중 오류 발생: {str(e)}")
        return image.convert('RGB')

def optimize_image(image: Image.Image) -> Image.Image:
    """
    이미지를 최적화하여 처리 속도를 향상시킵니다.
    """
    try:
        # 이미지 형식 변환
        image = convert_image_format(image)
        
        # 이미지 크기 최적화
        max_size = 600
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return image
    except Exception as e:
        logger.error(f"이미지 최적화 중 오류 발생: {str(e)}")
        return image

def extract_text_from_image(image: Image.Image) -> str:
    """
    이미지에서 텍스트를 추출합니다.
    """
    try:
        # 이미지 최적화
        optimized_image = optimize_image(image)
        
        # OCR 설정 (빠른 모드)
        custom_config = r'--oem 1 --psm 6 -l kor+eng'
        
        # OCR 수행
        text = pytesseract.image_to_string(optimized_image, config=custom_config)
        return text.strip()
    except Exception as e:
        logger.error(f"텍스트 추출 중 오류 발생: {str(e)}")
        return ""

def analyze_image_with_llm(image: Image.Image, prompt: str) -> str:
    """
    PIL Image 객체와 프롬프트를 받아 LLM으로 이미지를 분석하고 텍스트 설명을 반환합니다.
    """
    try:
        # 이미지 최적화
        optimized_image = optimize_image(image)
        
        # 이미지 바이트 변환
        img_byte_arr = io.BytesIO()
        optimized_image.save(img_byte_arr, format='JPEG', quality=85, optimize=True)
        img_bytes = img_byte_arr.getvalue()
        
        # 이미지 해시 생성
        image_hash = get_image_hash(img_bytes)
        
        # 캐시된 결과가 있는지 확인
        if image_hash in image_cache:
            return image_cache[image_hash]
        
        # 텍스트 추출
        extracted_text = extract_text_from_image(optimized_image)
        
        # base64로 인코딩
        base64_image = base64.b64encode(img_bytes).decode('utf-8')
        
        # 시스템 프롬프트
        system_prompt = "You are an AI that analyzes images and provides accurate translations."
        
        # 사용자 프롬프트
        enhanced_prompt = f"{prompt}\nExtracted text: {extracted_text}"
        
        # LangChain이 지원하는 형식으로 메시지 구성
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": enhanced_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}"
                    }
                ]
            )
        ]
        
        # llm_image 모델 사용
        response = llm_image.invoke(messages)
        result = response.content if hasattr(response, 'content') else str(response)
        
        # 결과 캐싱
        image_cache[image_hash] = result
        return result

    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        return f"이미지 분석 중 오류 발생: {str(e)}"