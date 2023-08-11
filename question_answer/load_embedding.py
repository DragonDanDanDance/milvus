from towhee import pipe, ops
import numpy as np
import common.milvus_v as mvar

insert_pipe = (
    pipe.input('id', 'question', 'answer')
        .map('question', 'vec', ops.text_embedding.dpr(model_name=mvar.get_model()))
        .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
        .map(('id', 'vec'), 'insert_status', ops.ann_insert.milvus_client(host=mvar.get_ip(), port=mvar.get_port(), collection_name=mvar.get_collection_name()))
        .output()
)

import csv
with open(mvar.get_q_a_path(), encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        insert_pipe(*row)
