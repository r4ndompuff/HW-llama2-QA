# 3.11.5
FROM python:3.11

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt
RUN mkdir ./app/sbert

#
#RUN pip install --upgrade pip
RUN curl -L https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf --output ./app/llama-2-7b-chat.Q4_K_M.gguf
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
RUN python -m nltk.downloader punkt averaged_perceptron_tagger

#
COPY ./app /code/app

# 
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--reload"]
