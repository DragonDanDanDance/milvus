#!/bin/bash

python create_milvus_collection.py
python load_embedding.py

echo "所有脚本已完成!"