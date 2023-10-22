from fastapi import FastAPI
from pydantic import BaseModel
import qdrant_client
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import UnstructuredPDFLoader, CSVLoader
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.schema.runnable import RunnablePassthrough
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
import os

rag_prompt_custom = PromptTemplate.from_template("""Используй следующие фрагменты документов для ответа на вопрос в конце.
Если ты не знаешь ответа, то просто скажи, что ты не знаешь, не пытайся придумать ответ.
Используй максимум 3 предложения и сохраняй ответ максимальное конкретным и ясным.
Всегда говори "Спасибо за Ваш вопрос!" в начале ответа. Ответ отдавай только на Русском языке.
Не говори номер документа, говори "По моей информации".
Контекст (фрагменты документов): {context}
Вопрос: {question}
Ответ:
""")


def filter_context(qdrant, context: str) -> str:
    context = [i.page_content for i in qdrant.max_marginal_relevance_search(context, k = 4, fetch_k = 5)]
    if len(context[0]) > 4000:
        return [context[0][:4000]] # Обрезаем самый вероятный контекст
    while sum([len(i) for i in context]) > 4_000:
        context = context[:-1]
    return ';\n'.join([f'{i}. {k}' for i, k in enumerate(context)])
    
    
def get_files():
    pdf_folder_path = "./app/content/"
    loaders_pdf = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn), 
                                 mode="elements", 
                                 strategy="fast") for fn in os.listdir(pdf_folder_path) if '.pdf' in fn]

    loaders_csv = [CSVLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path) if '.csv' in fn]

    loaders = loaders_pdf+loaders_csv
    all_documents = []
    for loader in loaders:
        raw_documents = loader.load()

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=2500,
            chunk_overlap=300,
            length_function=len,
        )
        documents = text_splitter.split_documents(raw_documents)
        all_documents.extend(documents)
    return all_documents

    
def get_qdrant_client():
    qdrant = Qdrant.from_documents(
        get_files(),
        SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", model_kwargs={'device': 'cpu'}),
        location=":memory:",  # Local mode with in-memory storage only
        #path="./code/vector_base",
        collection_name="vector_base",
    )
    return qdrant
    
    
def get_lama_response(question: str,
                     rag_prompt_custom,
                     qdrant) -> str:
    llm = LlamaCpp(
        model_path="./app/llama-2-7b-chat.Q4_K_M.gguf",
        temperature=0.75,
        max_tokens=5000,
        n_ctx=2048,
        top_p=1, 
        verbose=False
    )
    rag_chain = (
        {"context": lambda x: filter_context(qdrant, x), 
         "question": lambda x: x[:100] if len(x)>100 else x} 
        | rag_prompt_custom 
        | llm 
    )
    
    return rag_chain.invoke(question)
    

app = FastAPI()
qdrant = get_qdrant_client()


class Message(BaseModel):
    message: str
    user_id: str | None = None

@app.post("/message")
async def read_root(message: Message):
    answer = get_lama_response(message.message, rag_prompt_custom, qdrant)
    return answer
