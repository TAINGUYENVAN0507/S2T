#!/bin/bash -ex

export LD_LIBRARY_PATH=/home/brainx/anaconda3/envs/OCR/lib/python3.9/site-packages/nvidia/cublas/lib:/home/brainx/anaconda3/envs/OCR/lib/python3.9/site-packages/nvidia/cudnn/lib

uvicorn api:app --host 0.0.0.0 --port 8100 --ssl-keyfile /etc/nginx/ssl/connectedbrain.com.vn/privkey.pem --ssl-certfile /etc/nginx/ssl/connectedbrain.com.vn/fullchain.pem
