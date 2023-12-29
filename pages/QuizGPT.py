import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser, output_parser
from dotenv import load_dotenv
import os

load_dotenv()
# openai_api_key = os.getenv("OPENAI_API_KEY")

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    너는 대한민국의 초등학교 4학년 수학선생님이야.
    
    주어진 컨텍스트에만 의존해서 테스트용 질문을 5개 만들어.
    
    모든 질문은 4개의 answers을 갖고, 3개는 오답, 1개는 정답이어야 해.

    정답에는 (o) 표시를 해줘.
    
    모든 질문은 한국어로 진행해줘.
         
    질문 예시 : 
         
    질문 : 바다는 무슨색이야?
    answers : 빨간색|노란색|초록색|파란색(o)
         
    질문: 한국의 수도는?
    answers : 대구|부산|인천|서울(o)
         
    질문: 아바타 개봉날짜는?
    answers: 2007|2001|2009(o)|1998
         
    이제 네가 할 차례야.
         
    Context: {context}
""",
        )
    ]
)

questions_chain = {"context": format_docs} | questions_prompt | llm

formatting_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    너는 강력한 포맷 알고리즘이야.
     
    너는 질문과 답변을 JSON 포맷으로 만들어야해.
    (o)표시된 답변은 정답이야.
     
    질문 예시:
    
    질문 : 바다는 무슨색이야?
    answers : 빨간색|노란색|초록색|파란색(o)
         
    질문: 한국의 수도는?
    answers : 대구|부산|인천|서울(o)
         
    질문: 아바타 개봉날짜는?
    answers: 2007|2001|2009(o)|1998
     
    출력 예시:

    ```json
    {{ "질문들": [
            {{
                "질문": "바다는 무슨색이야?",
                "answers": [
                        {{
                            "answer": "빨간색",
                            "correct": false
                        }},
                        {{
                            "answer": "노란색",
                            "correct": false
                        }},
                        {{
                            "answer": "초록색",
                            "correct": false
                        }},
                        {{
                            "answer": "파란색",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "질문": "한국의 수도는?",
                "answers": [
                        {{
                            "answer": "대구",
                            "correct": false
                        }},
                        {{
                            "answer": "부산",
                            "correct": false
                        }},
                        {{
                            "answer": "인천",
                            "correct": false
                        }},
                        {{
                            "answer": "서울",
                            "correct": true
                        }}
                ]
            }},
                        {{
                "질문": "아바타 개봉날짜는?",
                "answers": [
                        {{
                            "answer": "2007",
                            "correct": false
                        }},
                        {{
                            "answer": "2001",
                            "correct": false
                        }},
                        {{
                            "answer": "2009",
                            "correct": true
                        }},
                        {{
                            "answer": "1998",
                            "correct": false
                        }}
                ]
            }},
        ]
     }}
    ```
    네 차례야!

    Questions: {context}

""",
        )
    ]
)

formatting_chain = formatting_prompt | llm


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs


with st.sidebar:
    docs = None
    topic = None
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)


if not docs:
    st.markdown(
        """
   QuizGPT.
    """
    )
else:
    response = run_quiz_chain(docs, topic if topic else file.name)
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print('-----------------------------------------')
    print(response)
    with st.form("questions_form"):
        for question in response["질문들"]:
            st.write(question["질문"])
            value = st.radio(
                "Select an option.",
                [answer["answer"] for answer in question["answers"]],
                index=None,
            )
            if {"answer": value, "correct": True} in question["answers"]:
                st.success("Correct!")
            elif value is not None:
                st.error("Wrong!")
        button = st.form_submit_button()