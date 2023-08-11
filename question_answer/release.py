import common.milvus_v as mvar
from pymilvus import Collection, connections
import gradio
from towhee import pipe, ops
import numpy as np
import pandas as pd

__df = pd.read_csv(mvar.get_q_a_path())
__df.head()
__id_answer = __df.set_index('id')['answer'].to_dict()

def main():
    # connect milvus
    connections.connect(host=mvar.get_ip(), port=mvar.get_port())
    collection = Collection(name=mvar.get_collection_name())
    collection.load()
    # launch gradio
    chatbot = gradio.Chatbot(color_map=("green", "gray"))
    interface = gradio.Interface(
        chat,
        ["text", "state"],
        [chatbot, "state"],
        allow_screenshot=False,
        allow_flagging="never",
    )
    interface.launch(server_name="0.0.0.0", inline=True, share=True)
    # TODO Pre-load

def chat(message, history):
    history = history or []
    ans_pipe = (
        pipe.input('question')
            .map('question', 'vec', ops.text_embedding.dpr(model_name=mvar.get_model()))
            .map('vec', 'vec', lambda x: x / np.linalg.norm(x, axis=0))
            .map('vec', 'res', ops.ann_search.milvus_client(host=mvar.get_ip(), port=mvar.get_port(), collection_name=mvar.get_collection_name(), limit=1))
            .map('res', 'answer', lambda x: [__id_answer[int(i[0])] for i in x])
            .output('question', 'answer')
    )

    response = ans_pipe(message).get()[1][0]
    history.append((message, response))
    return history, history

if __name__ == "__main__":
    main()
