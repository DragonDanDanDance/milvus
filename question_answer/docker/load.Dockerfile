FROM python:3.9.17

ENV MILVUS_IP=

ENV MILVUS_PORT=

ENV MILVUS_DATASET_PATH=

RUN mkdir -p /usr/local/towhee/model

RUN mkdir -p /usr/local/towhee/common

RUN mkdir -p /usr/local/towhee/dataset

WORKDIR /usr/local/towhee

ADD ./create_milvus_collection.py .

ADD ./load_embedding.py .

ADD ./common/milvus_v.py ./common

ADD ./dataset/question_answer.csv ./dataset

ADD ./docker/entrypoint.sh .

ADD ./requirements.txt .

RUN python -m pip install -r requirements.txt

ENTRYPOINT ["sh", "./entrypoint.sh"]