# Univera RAG Chatbot

## 개요
Univera 회사 정보 기반 RAG (Retrieval-Augmented Generation) 챗봇입니다. 하이브리드 검색(벡터 검색 + BM25)과 GPT-4o-mini를 활용하여 정확한 답변을 제공합니다.

## 주요 기능
- 🔍 **하이브리드 검색**: 벡터 검색과 BM25 검색을 결합
- 🤖 **AI 답변 생성**: GPT-4o-mini 기반 한국어 답변
- 📊 **실시간 검색 통계**: 검색 결과 및 성능 모니터링
- ⚙️ **설정 가능**: 검색 가중치 및 파라미터 조정
- 📱 **반응형 UI**: Streamlit 기반 사용자 친화적 인터페이스

## 사용된 기술
- **임베딩 모델**: `intfloat/multilingual-e5-base` (다국어 지원)
- **벡터 데이터베이스**: Pinecone
- **언어 모델**: GPT-4o-mini (OpenAI)
- **검색 방법**: 벡터 검색 + BM25 하이브리드
- **프론트엔드**: Streamlit

## 설치 및 설정

### 1. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정
`.env.example`을 복사해서 `.env` 파일을 만들고 실제 API 키를 입력하세요:
```bash
cp .env.example .env
```
그리고 `.env` 파일에서 `your_openai_api_key_here` 등을 실제 API 키로 교체하세요.

### 3. Streamlit Secrets 설정 (Cloud 배포용)
`.streamlit/secrets.toml.example`을 참고하여 Streamlit Cloud의 Secrets 설정에 실제 API 키를 입력하세요.

## 실행 방법

### 로컬 실행
```bash
streamlit run app.py
```

### Streamlit Cloud 배포
1. **GitHub에 코드 업로드**
   ```bash
   git add .
   git commit -m "Add Univera RAG Chatbot"
   git push origin main
   ```

2. **Streamlit Cloud에서 앱 생성**
   - [share.streamlit.io](https://share.streamlit.io) 접속
   - GitHub 계정으로 로그인
   - "New app" → Repository, Branch(main), Main file(app.py) 선택

3. **Secrets 설정** (중요!)
   - Streamlit Cloud 대시보드 → Settings → Secrets
   - `.streamlit/secrets.toml.example` 내용을 복사하고 실제 API 키로 교체해서 붙여넣기

4. **자동 배포 완료**

## 사용법
1. 웹 브라우저에서 앱 접속
2. 사이드바에서 검색 설정 조정 (선택사항)
3. 채팅 인터페이스에서 Univera 관련 질문 입력
4. AI 답변 및 참고 문서 확인

## 주요 설정
- **벡터 검색 가중치**: 의미적 유사도 기반 검색 비중
- **BM25 검색 가중치**: 키워드 기반 검색 비중  
- **최종 검색 결과 개수**: AI 답변 생성에 사용할 문서 수
- **최대 답변 토큰**: 생성되는 답변의 최대 길이

## 시스템 아키텍처
1. **사용자 질문 입력**
2. **하이브리드 검색 수행**
   - 벡터 검색: Pinecone에서 의미적 유사 문서 검색
   - BM25 검색: 키워드 기반 문서 검색
3. **결과 통합 및 랭킹**
4. **GPT-4o-mini로 답변 생성**
5. **답변 및 참고 문서 표시**

## 문제 해결
- **API 키 오류**: 환경 변수나 Secrets 설정 확인
- **Pinecone 연결 오류**: API 키와 인덱스 이름 확인
- **검색 결과 없음**: 데이터베이스에 문서가 업로드되었는지 확인

## 개발자 정보
- 기반 노트북: `RAG_univera_v.250905.ipynb`
- 임베딩 모델: multilingual-e5-base (768차원)
- 벡터 DB: Pinecone (rag-univera-pinecone-db)