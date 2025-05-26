from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# Ollama 서버 주소 (필요시 변경)
OLLAMA_BASE_URL = "http://localhost:11434"
TOKEN_CONTEXT_LENGTH = 4096

# 모델 이름 정의 (Ollama에 pull된 모델명과 일치해야 함)
MODEL_QWEN = "qwen3:latest" # 일반 질문 메인
MODEL_EMBEDDING = "nomic-embed-text:latest" # 임베딩용
MODEL_LLAVA = "llava:7b" # 이미지, pdf
MODEL_LLAMA = "llama3.2:latest" # 복잡한 추론/ 이미지 서브
MODEL_DEEPSEEK = "deepseek-r1:latest" # 코딩/수학

def get_llm(model_name: str, temperature: float = 0.1):
    """지정된 모델 이름으로 ChatOllama 인스턴스를 반환합니다."""
    return ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=model_name,
        temperature=temperature,
        num_ctx=TOKEN_CONTEXT_LENGTH, # 컨텍스트 길이 설정
    )

llm_general = get_llm(MODEL_QWEN) # 일반 질문용
llm_embedding = get_llm(MODEL_EMBEDDING) # 임베딩용
llm_image = get_llm(MODEL_LLAVA) # 이미지/PDF 분석용
llm_reasoning = get_llm(MODEL_LLAMA) # 복잡한 추론/이미지 서브
llm_coding = get_llm(MODEL_DEEPSEEK) # 코딩/수학용

# 임베딩 모델
embeddings = OllamaEmbeddings(
    base_url=OLLAMA_BASE_URL,
    model=MODEL_EMBEDDING
)

# 사용 가능한 모델 목록 (라우팅 로직에서 사용)
AVAILABLE_MODELS = {
    "general": llm_general,
    "image_analysis": llm_image,
    "reasoning": llm_reasoning,
    "coding_math": llm_coding
}