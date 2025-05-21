from typing import List, Tuple, TypedDict, Annotated, Sequence, Literal, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langgraph.graph import StateGraph, END
import operator

from .llm_config import AVAILABLE_MODELS, llm_general, llm_coding, llm_reasoning
from .rag_handler import get_rag_retriever, query_rag
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
        # 이미지와 관련된 질문인지, 아니면 별도 질문인지 판단
        # 여기서는 간단히 이미지가 있으면 이미지 분석으로 라우팅
        return {"selected_agent": "image_analysis_route"}

    # RAG 사용 여부 판단 (예: "내 문서에서 찾아줘", "PDF 내용 요약")
    if "pdf" in query or "문서" in query or "내 파일" in query or "내 자료" in query:
        return {"selected_agent": "rag"}

    # 키워드 기반 라우팅 (더 정교한 분류기 LLM 사용 가능)
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

    # 이미지와 관련된 질문이 있을 경우, 해당 질문을 프롬프트로 사용
    # 이미지만 있고 텍스트 질문이 없다면, 일반적인 분석 프롬프트 사용
    analysis_prompt = query if query else "이 이미지에 대해 설명해주세요."
    
    print(f"Analyzing image with prompt: {analysis_prompt}")
    analysis_result = analyze_image_with_llm(image, analysis_prompt)
    
    # 이미지 분석 결과를 다음 단계에서 활용하기 위해 상태에 저장
    # 만약 이미지 분석 자체가 최종 답변이라면 output_message에 바로 할당
    # 여기서는 분석 결과를 다음 LLM 호출의 컨텍스트로 활용하도록 설계
    return {
        "image_analysis_result": analysis_result,
        "intermediate_steps": state.get("intermediate_steps", []) + [f"Image analysis result: {analysis_result[:200]}..."]
    }

def rag_node(state: AgentState) -> AgentState:
    """RAG를 사용하여 컨텍스트를 검색하고 상태에 저장합니다."""
    query = state["input_query"]
    print(f"Performing RAG search for: {query}")
    relevant_docs = query_rag(query, k=3)
    
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
    if not llm: # 라우팅 오류 또는 "rag", "image_analysis_route" 같은 중간 단계
        # 이 경우, image_analysis_node나 rag_node에서 이미 필요한 정보를 처리했거나,
        # 다음 라우팅에서 적절한 LLM으로 연결되어야 함.
        # 여기서는 image_analysis_result나 rag_context를 사용해 일반 LLM으로 질문을 다시 구성
        llm = llm_general 
        
        # 컨텍스트를 포함하여 새로운 질문 구성
        if image_analysis_context and rag_context:
            effective_query = f"이미지 분석 결과: {image_analysis_context}\n\n문서 내용: {rag_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        elif image_analysis_context:
            effective_query = f"이미지 분석 결과: {image_analysis_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        elif rag_context:
            effective_query = f"문서 내용: {rag_context}\n\n위 정보를 바탕으로 다음 질문에 답해주세요: {query}"
        else: # 컨텍스트 없이 원래 질문 사용
            effective_query = query
    else: # coding_math, reasoning, general
        effective_query = query
        if image_analysis_context: # 이미지 분석 결과가 있다면 프롬프트에 추가
            effective_query = f"참고 이미지 분석: {image_analysis_context}\n\n질문: {query}"
        if rag_context: # RAG 결과가 있다면 프롬프트에 추가
            effective_query = f"참고 문서: {rag_context}\n\n질문: {effective_query}"


    print(f"Calling LLM ({agent_name if agent_name in AVAILABLE_MODELS else 'general_fallback'}) with query: {effective_query[:200]}...")

    # 프롬프트 템플릿 구성
    # SystemMessage를 첫 번째로, 그 다음 MessagesPlaceholder, 마지막으로 HumanMessage
    prompt_messages = [SystemMessage(content="You are a helpful AI assistant.")]
    if history:
        prompt_messages.append(MessagesPlaceholder(variable_name="chat_history_placeholder"))
    prompt_messages.append(HumanMessage(content=effective_query))
    
    prompt = ChatPromptTemplate.from_messages(prompt_messages)
    
    chain = prompt | llm
    
    # LangChain Expression Language (LCEL)을 사용하여 호출
    # 히스토리 객체는 BaseMessage 리스트여야 함
    response = chain.invoke({"chat_history_placeholder": history, "input": effective_query})
    
    output_message = response.content if hasattr(response, 'content') else str(response)
    
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
workflow.add_node("final_llm_call", llm_call_node) # 이미지/RAG 후 최종 LLM 호출

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
    else: # general
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

# 이미지 분석 후 또는 RAG 후의 라우팅
def decide_after_preprocessing(state: AgentState):
    # 이미지 분석과 RAG가 모두 완료되었거나, 둘 중 하나만 필요했던 경우
    # 이제 최종 LLM 호출로 이동
    return "final_llm_call"

workflow.add_edge("image_analyzer", "final_llm_call") # 이미지 분석 후에는 항상 최종 LLM 호출
workflow.add_edge("rag_retriever", "final_llm_call")  # RAG 후에는 항상 최종 LLM 호출

# 단순 에이전트 호출 후 종료
workflow.add_edge("coding_math_agent", END)
workflow.add_edge("reasoning_agent", END)
workflow.add_edge("general_agent", END)
workflow.add_edge("final_llm_call", END) # 최종 LLM 호출 후 종료

# 그래프 컴파일
app_graph = workflow.compile()

# 그래프 실행 함수
def run_graph(query: str, chat_history: List[Tuple[str, str]], image_pil: Optional[Image.Image] = None):
    # Gradio 히스토리를 LangChain 메시지 형식으로 변환
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
        "selected_agent": "general", # 기본값, 라우터에서 변경됨
        "output_message": None,
        "intermediate_steps": []
    }
    
    # stream() 대신 invoke() 사용 (Gradio와의 호환성 및 단순성)
    final_state = app_graph.invoke(initial_state)
    
    return final_state.get("output_message", "죄송합니다. 답변을 생성하지 못했습니다.")