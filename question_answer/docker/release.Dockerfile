FROM python:3.9.17

ENV MILVUS_IP=

ENV MILVUS_PORT=

ENV MILVUS_DATASET_PATH=

RUN mkdir -p /usr/local/towhee/model

RUN mkdir -p /usr/local/towhee/common

RUN mkdir -p /usr/local/towhee/dataset

WORKDIR /usr/local/towhee

ADD ./release.py .

ADD ./common/milvus_v.py ./common

ADD ./dataset/question_answer.csv ./dataset

ADD ./requirements.txt .

RUN python -m pip install -r requirements.txt

ENTRYPOINT python release.py

EXPOSE 7860