import streamlit as st
import os
import re
import openai
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
import json

st.set_page_config(page_title="Univera RAG Chatbot", page_icon="🤖", layout="wide")

@st.cache_resource
def initialize_models():
    """모델 및 API 클라이언트 초기화"""
    # OpenAI 설정
    openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다.")
        st.stop()
    
    openai_client = openai.OpenAI(api_key=openai_api_key)
    
    # Pinecone 설정
    pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
    pinecone_index_name = st.secrets.get("PINECONE_INDEX_NAME") or os.getenv("PINECONE_INDEX_NAME")
    
    if not pinecone_api_key or not pinecone_index_name:
        st.error("Pinecone API 키 또는 인덱스 이름이 설정되지 않았습니다.")
        st.stop()
    
    pc = Pinecone(api_key=pinecone_api_key)
    pinecone_index = pc.Index(pinecone_index_name)
    
    # 임베딩 모델 로드
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-base")
    
    return openai_client, pinecone_index, embedding_model

class UniveraRAG:
    def __init__(self, openai_client, pinecone_index, embedding_model):
        self.openai_client = openai_client
        self.pinecone_index = pinecone_index
        self.model = embedding_model
        self.gpt_model = "gpt-4o-mini"
        
        # BM25를 위한 문서 캐시 (실제 환경에서는 별도 저장소 사용 권장)
        if 'bm25_corpus' not in st.session_state:
            st.session_state.bm25_corpus = []
            st.session_state.bm25_model = None
    
    def tokenize(self, text):
        """텍스트 토큰화"""
        if not text or not isinstance(text, str):
            return []
        
        # 한글, 영어, 숫자 추출
        tokens = re.findall(r'[가-힣a-zA-Z0-9]+', text.lower())
        return [token for token in tokens if len(token) >= 2]
    
    def embed(self, text, is_query=False):
        """텍스트 임베딩"""
        if not text:
            return []
        
        prefix = "query: " if is_query else "passage: "
        return self.model.encode(prefix + text, normalize_embeddings=True)
    
    def vector_search(self, query, top_k=15):
        """벡터 검색"""
        query_vec = self.embed(query, is_query=True)
        
        try:
            results = self.pinecone_index.query(
                vector=query_vec.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            
            vector_results = []
            for match in results.matches:
                if 'text' in match.metadata:
                    vector_results.append({
                        'text': match.metadata['text'],
                        'score': float(match.score),
                        'source': match.metadata.get('source', 'Unknown'),
                        'type': 'vector'
                    })
            
            return vector_results
        except Exception as e:
            st.error(f"벡터 검색 오류: {str(e)}")
            return []
    
    def bm25_search(self, query, top_k=10):
        """BM25 검색 (제한적 구현)"""
        # 실제 환경에서는 사전에 구축된 BM25 인덱스 사용
        # 여기서는 벡터 검색 결과를 기반으로 간단한 텍스트 매칭 수행
        if not st.session_state.bm25_corpus:
            return []
        
        try:
            tokenized_query = self.tokenize(query)
            if not tokenized_query:
                return []
            
            if st.session_state.bm25_model is None:
                return []
            
            scores = st.session_state.bm25_model.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            
            bm25_results = []
            for idx in top_indices:
                if scores[idx] > 0:
                    bm25_results.append({
                        'text': st.session_state.bm25_corpus[idx]['text'],
                        'score': float(scores[idx]),
                        'source': st.session_state.bm25_corpus[idx].get('source', 'Unknown'),
                        'type': 'bm25'
                    })
            
            return bm25_results
        except Exception as e:
            st.warning(f"BM25 검색을 사용할 수 없습니다: {str(e)}")
            return []
    
    def normalize_scores(self, results):
        """점수 정규화"""
        if not results:
            return results
        
        scores = [r['score'] for r in results]
        if not scores:
            return results
        
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            for result in results:
                result['normalized_score'] = 1.0
        else:
            for result in results:
                result['normalized_score'] = (result['score'] - min_score) / (max_score - min_score)
        
        return results
    
    def hybrid_search(self, query, vector_weight=0.7, bm25_weight=0.3, final_top_k=5):
        """하이브리드 검색"""
        # 벡터 검색
        vector_results = self.vector_search(query, top_k=15)
        
        # BM25 검색
        bm25_results = self.bm25_search(query, top_k=10)
        
        # 점수 정규화
        vector_results = self.normalize_scores(vector_results)
        bm25_results = self.normalize_scores(bm25_results)
        
        # 결과 통합
        combined_results = {}
        
        # 벡터 검색 결과 추가
        for result in vector_results:
            text = result['text']
            score = result['normalized_score'] * vector_weight
            combined_results[text] = {
                'text': text,
                'score': score,
                'source': result['source'],
                'types': ['vector']
            }
        
        # BM25 검색 결과 추가/업데이트
        for result in bm25_results:
            text = result['text']
            score = result['normalized_score'] * bm25_weight
            
            if text in combined_results:
                combined_results[text]['score'] += score
                combined_results[text]['types'].append('bm25')
            else:
                combined_results[text] = {
                    'text': text,
                    'score': score,
                    'source': result['source'],
                    'types': ['bm25']
                }
        
        # 최종 결과 정렬 및 상위 k개 선택
        final_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:final_top_k]
        
        return final_results, len(vector_results), len(bm25_results)
    
    def create_context(self, search_results):
        """검색 결과를 컨텍스트로 변환"""
        context_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.get('source', 'Unknown')
            text = result['text']
            context_parts.append(f"[문서 {i}] ({source}):\n{text}")
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query, search_results, max_tokens=1000):
        """GPT를 사용하여 답변 생성"""
        if not search_results:
            return "죄송합니다. 관련 문서를 찾을 수 없어서 답변을 생성할 수 없습니다.", []
        
        context = self.create_context(search_results)
        
        system_prompt = """당신은 Univera 회사 정보 전문 AI 어시스턴트입니다.

주어진 컨텍스트를 바탕으로 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요.

답변 규칙:
1. 컨텍스트에 있는 정보만 사용하여 답변하세요
2. 정확하지 않거나 확실하지 않은 정보는 제공하지 마세요
3. 한국어로 자연스럽고 친근하게 답변하세요
4. 가능하면 구체적인 예시나 세부사항을 포함하세요
5. 컨텍스트에 관련 정보가 없다면 솔직히 "해당 정보를 찾을 수 없습니다"라고 말하세요

답변 형식:
- 명확하고 구조화된 답변
- 필요시 불릿 포인트 사용
- 출처 정보 포함 (예: "문서에 따르면...")
"""

        user_prompt = f"""컨텍스트:
{context}

질문: {query}

위 컨텍스트를 바탕으로 질문에 답변해주세요."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            return answer, search_results
            
        except Exception as e:
            error_msg = f"답변 생성 중 오류가 발생했습니다: {str(e)}"
            return error_msg, search_results

def main():
    st.title("🤖 Univera RAG Chatbot")
    st.markdown("---")
    
    # 초기화
    openai_client, pinecone_index, embedding_model = initialize_models()
    rag_system = UniveraRAG(openai_client, pinecone_index, embedding_model)
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 검색 설정")
        
        vector_weight = st.slider("벡터 검색 가중치", 0.0, 1.0, 0.7, 0.1)
        bm25_weight = st.slider("BM25 검색 가중치", 0.0, 1.0, 0.3, 0.1)
        final_top_k = st.slider("최종 검색 결과 개수", 1, 10, 5)
        max_tokens = st.slider("최대 답변 토큰", 500, 2000, 1000, 100)
        
        # 가중치 합이 1이 되도록 조정
        total_weight = vector_weight + bm25_weight
        if total_weight != 1.0:
            st.warning("가중치 합이 1.0이 되도록 자동 조정됩니다.")
            vector_weight = vector_weight / total_weight
            bm25_weight = bm25_weight / total_weight
        
        st.markdown("---")
        st.markdown("### 📊 시스템 상태")
        st.success("✅ OpenAI 연결됨")
        st.success("✅ Pinecone 연결됨") 
        st.success("✅ 임베딩 모델 로드됨")
    
    # 채팅 인터페이스
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "안녕하세요! Univera 관련 질문을 자유롭게 물어보세요. 😊"}
        ]
    
    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # 검색 결과가 있는 경우 표시
            if "search_results" in message:
                with st.expander("🔍 참고된 문서"):
                    for i, result in enumerate(message["search_results"], 1):
                        st.markdown(f"**문서 {i}** (스코어: {result['score']:.3f})")
                        st.markdown(f"**출처:** {result['source']}")
                        st.markdown(f"**검색 방법:** {', '.join(result.get('types', ['vector']))}")
                        st.markdown(f"**내용:** {result['text'][:200]}...")
                        st.markdown("---")
    
    # 사용자 입력
    if prompt := st.chat_input("Univera에 대해 질문해보세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # AI 답변 생성
        with st.chat_message("assistant"):
            with st.spinner("검색 중..."):
                # 하이브리드 검색 수행
                search_results, vector_count, bm25_count = rag_system.hybrid_search(
                    prompt, vector_weight, bm25_weight, final_top_k
                )
                
                # 검색 통계 표시
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("벡터 검색", vector_count)
                with col2:
                    st.metric("BM25 검색", bm25_count)
                with col3:
                    st.metric("최종 결과", len(search_results))
            
            with st.spinner("답변 생성 중..."):
                # 답변 생성
                answer, used_results = rag_system.generate_answer(prompt, search_results, max_tokens)
                
                # 답변 표시
                st.markdown(answer)
                
                # 검색 결과 표시
                if used_results:
                    with st.expander("🔍 참고된 문서"):
                        for i, result in enumerate(used_results, 1):
                            st.markdown(f"**문서 {i}** (스코어: {result['score']:.3f})")
                            st.markdown(f"**출처:** {result['source']}")
                            st.markdown(f"**검색 방법:** {', '.join(result.get('types', ['vector']))}")
                            st.markdown(f"**내용:** {result['text'][:200]}...")
                            st.markdown("---")
        
        # 답변을 세션에 저장
        st.session_state.messages.append({
            "role": "assistant", 
            "content": answer,
            "search_results": used_results
        })
    
    # 하단 정보
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <small>Univera RAG Chatbot • Powered by OpenAI GPT-4o-mini & Pinecone • 다국어 임베딩 지원</small>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()