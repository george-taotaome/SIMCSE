# -*- coding: utf-8 -*-

import logging
import argparse
from SimCSERetrieval import SimCSERetrieval

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train_file", type=str, default="./data/all.txt", help="train text file")
    parser.add_argument("--pretrained", type=str, default="hfl/chinese-bert-wwm-ext", help="huggingface pretrained model")
    parser.add_argument("--model", type=str, default="./model/epoch_16-batch_6000-loss_0.029720", help="model file path")
    parser.add_argument("--max_length", type=int, default=100, help="sentence max length")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--device", type=str, default="cuda", help="device")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    logging.info("Load model")
    simcse = SimCSERetrieval(args.train_file, args.pretrained, args.model, args.batch_size, args.max_length, args.device)

    logging.info("Sentences to vectors")
    simcse.encode_file_and_index()

    logging.info("Build faiss index")
    simcse.build_index(nlist=1024)

if __name__ == "__main__":
    log_fmt = "%(asctime)s|%(name)s|%(levelname)s|%(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    main()
