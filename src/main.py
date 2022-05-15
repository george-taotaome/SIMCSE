# -*- coding: utf-8 -*-

from fastapi import FastAPI
import dto
from SimCSERetrieval import SimCSERetrieval
import faiss
import uvicorn
import multiprocessing
import logging
import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="./data/all.txt", help="train text file")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model", type=str, default="./model/epoch_16-batch_6000-loss_0.029720", help="model file")
    parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--device", type=str, default="cpu", help="device")
    parser.add_argument("--index", type=str, default="./data/simcse.trained.index", help="faiss index file")
    parser.add_argument("--workers", type=int, default=multiprocessing.cpu_count(), help="workers count")
    args = parser.parse_args()
    return args

args = parse_args()
logging.info("Load model")
simcse = SimCSERetrieval(args.train_file, args.pretrained, args.model, args.batch_size, args.max_length, args.device)
logging.info("Load Sentences")
simcse.encode_file()
logging.info("Load faiss index")
simcse.index = faiss.read_index(args.index)

app = FastAPI()

"""
语义匹配查询
"""
@app.post("/sim")
async def read_root(sim_dto: dto.Queue):
    try:
        res = simcse.sim_query(sim_dto.text, sim_dto.limit)
    except Exception as ex:
        logging.error(ex)
        return {"result": False, "message": "error"}
    try:
        return {"result": res, "message": "ok"}
    except Exception as ex:
        logging.error(ex)

if __name__ == '__main__':
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logging.info("Start service")
    uvicorn.run(app="main:app", host="0.0.0.0", port=8000, workers=args.workers)
