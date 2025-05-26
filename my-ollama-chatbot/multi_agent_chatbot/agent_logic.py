from typing import List, Tuple, TypedDict, Annotated, Sequence, Literal, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
import operator

from .llm_config import AVAILABLE_MODELS, llm_general, llm_coding, llm_reasoning, llm_image
from .rag_handler import get_relevant_documents, query_pdf_content
from .image_handler import analyze_image_with_llm
from PIL import Image


# --- Agent State ---
class AgentState(TypedDict):
    input_query: str
    chat_history: Annotated[Sequence[BaseMessage], operator.add]
    image_data: Optional[Image.Image] # PIL Image
    image_analysis_result: Optional[str]
    rag_context: Optional[str]
    selected_agent: Literal["coding_math", "reasoning", "general", "rag", "image_analysis_route"]
    output_message: Optional[str]
    intermediate_steps: list # 디버깅용

# --- Nodes ---
def route_query_node(state: AgentState) -> AgentState:
    """쿼리 유형에 따라 다음 노드를 결정합니다."""
    query = state["input_query"].lower()
    image_data = state["image_data"]
    
    # 이미지 분석이 우선순위가 높을 경우
    if image_data:
        return {"selected_agent": "image_analysis_route"}

    # RAG 사용 여부 판단
    if any(kw in query for kw in ["pdf", "문서", "내 파일", "내 자료", "찾아줘", "검색", "요약"]):
        return {"selected_agent": "rag"}

    # 키워드 기반 라우팅
    if any(kw in query for kw in ["코드", "코딩", "프로그래밍", "알고리즘", "수학", "계산", "풀어줘"]):
        return {"selected_agent": "coding_math"}
    elif any(kw in query for kw in ["추론", "분석", "설명해줘", "왜", "어떻게 생각해"]):
        return {"selected_agent": "reasoning"}
    else:
        return {"selected_agent": "general"}

def image_analysis_node(state: AgentState) -> AgentState:
    """이미지를 분석하고 결과를 상태에 저장합니다."""
    image = state["image_data"]
    query = state["input_query"]
    
    if not image:
        return {"output_message": "이미지 분석을 요청했지만 이미지가 제공되지 않았습니다.", "image_analysis_result": None}

    analysis_prompt = query if query else "이 이미지에 대해 설명해주세요."
    
    print(f"Analyzing image with prompt: {analysis_prompt}")
    analysis_result = analyze_image_with_llm(image, analysis_prompt)
    
    return {
        "image_analysis_result": analysis_result,
        "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis result: {analysis_result[:200]}..."]
    }

def rag_node(state: AgentState) -> AgentState:
    """RAG를 사용하여 컨텍스트를 검색하고 상태에 저장합니다."""
    query = state["input_query"]
    print(f"Performing RAG search for: {query}")
    
    # get_relevant_documents 함수 사용
    relevant_docs = get_relevant_documents(query, k=3)
    
    if not relevant_docs:
        context = "관련 정보를 찾을 수 없습니다."
    else:
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    print(f"RAG Context (first 200 chars): {context[:200]}")
    return {
        "rag_context": context,
        "intermediate_steps": state.get("intermediate_steps", []) + [f"RAG context retrieved: {context[:200]}..."]
    }

def llm_call_node(state: AgentState) -> AgentState:
    """선택된 에이전트(LLM)를 호출하고 응답을 생성합니다."""
    agent_name = state["selected_agent"]
    query = state["input_query"]
    history = state["chat_history"]
    rag_context = state.get("rag_context")
    image_analysis_context = state.get("image_analysis_result")

    llm = AVAILABLE_MODELS.get(agent_name)
    if not llm:
        llm = llm_general 
        model_name = "qwen3:latest"
        
        if image_analysis_context and rag_context:
            effective_query = f"이미지 분석 결과: {image_analysis_context}\n\n문서 내용: {rag_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        elif image_analysis_context:
            effective_query = f"이미지 분석 결과: {image_analysis_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        elif rag_context:
            effective_query = f"문서 내용: {rag_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        else:
            effective_query = query
    else:
        effective_query = query
        if image_analysis_context:
            effective_query = f"참고 이미지 분석: {image_analysis_context}\n\n질문: {query}"
        if rag_context:
            effective_query = f"참고 문서: {rag_context}\n\n질문: {effective_query}"
        
        if agent_name == "coding_math":
            model_name = "deepseek-r1:latest"
        elif agent_name == "reasoning":
            model_name = "llama3.2:latest"
        elif agent_name == "general":
            model_name = "qwen3:latest"
        elif agent_name == "image_analysis":
            model_name = "llava:7b"
        else:
            model_name = "unknown"

    print(f"Calling LLM ({agent_name if agent_name in AVAILABLE_MODELS else 'general_fallback'}) with query: {effective_query[:200]}...")

    prompt_messages = [SystemMessage(content="You are a helpful AI assistant.")]
    if history:
        prompt_messages.append(MessagesPlaceholder(variable_name="chat_history_placeholder"))
    prompt_messages.append(HumanMessage(content=effective_query))
    
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    chain = prompt | llm
    
    response = chain.invoke({"chat_history_placeholder": history, "input": effective_query})
    
    output_message = response.content if hasattr(response, 'content') else str(response)
    output_message = f"[사용 모델: {model_name}]\n\n{output_message}"
    
    return {"output_message": output_message}


# --- Graph Definition ---
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("query_router", route_query_node)
workflow.add_node("image_analyzer", image_analysis_node)
workflow.add_node("rag_retriever", rag_node)
workflow.add_node("coding_math_agent", llm_call_node)
workflow.add_node("reasoning_agent", llm_call_node)
workflow.add_node("general_agent", llm_call_node)
workflow.add_node("final_llm_call", llm_call_node)

# 엣지 설정
workflow.set_entry_point("query_router")

def decide_next_step_after_routing(state: AgentState):
    if state["selected_agent"] == "image_analysis_route":
        return "image_analyzer"
    elif state["selected_agent"] == "rag":
        return "rag_retriever"
    elif state["selected_agent"] == "coding_math":
        return "coding_math_agent"
    elif state["selected_agent"] == "reasoning":
        return "reasoning_agent"
    else:
        return "general_agent"

workflow.add_conditional_edges(
    "query_router",
    decide_next_step_after_routing,
    {
        "image_analyzer": "image_analyzer",
        "rag_retriever": "rag_retriever",
        "coding_math_agent": "coding_math_agent",
        "reasoning_agent": "reasoning_agent",
        "general_agent": "general_agent",
    }
)

def decide_after_preprocessing(state: AgentState):
    return "final_llm_call"

workflow.add_edge("image_analyzer", "final_llm_call")
workflow.add_edge("rag_retriever", "final_llm_call")

workflow.add_edge("coding_math_agent", END)
workflow.add_edge("reasoning_agent", END)
workflow.add_edge("general_agent", END)
workflow.add_edge("final_llm_call", END)

# 그래프 컴파일
app_graph = workflow.compile()

# 그래프 실행 함수
def run_graph(query: str, chat_history: List[Tuple[str, str]], image_pil: Optional[Image.Image] = None):
    lc_history = []
    for human, ai in chat_history:
        lc_history.append(HumanMessage(content=human))
        lc_history.append(AIMessage(content=ai))

    initial_state: AgentState = {
        "input_query": query,
        "chat_history": lc_history,
        "image_data": image_pil,
        "image_analysis_result": None,
        "rag_context": None,
        "selected_agent": "general",
        "output_message": None,
        "intermediate_steps": []
    }
    
    final_state = app_graph.invoke(initial_state)
    
    return final_state.get("output_message", "죄송합니다. 답변을 생성하지 못했습니다.")