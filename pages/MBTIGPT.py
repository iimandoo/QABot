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

class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="MBTI GPT",
)

st.title("MBTI GPT")

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
    너는 심리학과 선생님이야.
    
    주어진 컨텍스트에만 의존해서 MBTI를 테스트해볼 수 있는 질문을 1개 만들어.
    
    모든 질문은 맞습니다, 아닙니다 의 2가지답변만 받을 수 있어.
    
    맞습니다 답변인 경우 해당 MBTI 를 보여줘.
    
    모든 질문은 한국어로 진행해줘.
         
    질문 예시 : 
         
    질문 : 해야될 것은 반드시 하고야 마는 책임감이 강한 사람입니까?
    answers : 맞습니다|아닙니다
         
    질문: 곧 움직일 준비가 되어있는 행동파입니까?
    answers : 맞습니다|아닙니다
         
    질문: 의무감이 투철합니까?
    answers : 맞습니다|아닙니다
         
         
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
     
    질문 예시:
    
    질문 : 해야될 것은 반드시 하고야 마는 책임감이 강한 사람입니까?
    answers : 맞습니다|아닙니다
         
    질문: 곧 움직일 준비가 되어있는 행동파입니까?
    answers : 맞습니다|아닙니다

    질문: 의무감이 투철합니까?
    answers : 맞습니다|아닙니다
         
    출력 예시:

    ```json
    {{ "질문들": [
            {{
                "질문": "해야될 것은 반드시 하고야 마는 책임감이 강한 사람입니까?",
                "answers": [
                        {{
                            "answer": "맞습니다",
                            "설명": "ISTJ(내향성 감각형)"
                        }},
                        {{
                            "answer": "아닙니다",
                            "설명": ""
                        }},
                ]
            }},
                        {{
                "질문": "곧 움직일 준비가 되어있는 행동파입니까?",
                "answers": [
                        {{
                            "answer": "맞습니다",
                            "설명": " ISTP(내향성 사고형)"
                        }},
                        {{
                            "answer": "아닙니다",
                            "설명": ""
                        }},
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
        chunk_size=400,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic):
    chain = {"context": questions_chain} | formatting_chain | output_parser
    return chain.invoke(_docs)


with st.sidebar:
    docs = None
    topic = None
    
    file = st.file_uploader(
        "Upload a .docx , .txt or .pdf file",
        type=["pdf", "txt", "docx"],
    )
    if file:
        docs = split_file(file)



if docs:
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
            if {"answer": value, "설명": ""} in question["answers"]:
                st.success("")
            elif value is not None:
                st.error(value)
        button = st.form_submit_button()