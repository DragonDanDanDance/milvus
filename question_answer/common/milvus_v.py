import os

milvus_ip: str = os.environ.get('MILVUS_IP')
milvus_port: str = os.environ.get('MILVUS_PORT')
milvus_dataset_path: str = os.environ.get('MILVUS_DATASET_PATH')

def get_ip():
    if milvus_ip is None or len(str(milvus_ip)) == 0:
       return '127.0.0.1'
    return str(milvus_ip)

def get_port():
    if milvus_port is None or len(str(milvus_port)) == 0:
       return '19530'
    return str(milvus_port)

def get_q_a_path():
    if milvus_dataset_path is None or len(str(milvus_dataset_path)) == 0:
       return './dataset/question_answer.csv'
    return str(milvus_dataset_path)

def get_collection_name():
    return 'question_answer'

def get_collection_dim():
    return 768

def get_model():
    return './model/dpr-ctx_encoder-single-nq-base'

