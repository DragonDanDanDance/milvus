import os

milvus_ip: str = os.environ.get('MILVUS_IP')
milvus_port: str = os.environ.get('MILVUS_PORT')
milvus_dataset_path: str = os.environ.get('MILVUS_DATASET_PATH')

def get_ip():
    return milvus_ip or '127.0.0.1'

def get_port():
    return milvus_port or '19530'

def get_q_a_path():
    return milvus_dataset_path or './dataset/question_answer.csv'

def get_collection_name():
    return 'question_answer'

def get_collection_dim():
    return 768

def get_model():
    return './model/dpr-ctx_encoder-single-nq-base'

