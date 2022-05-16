# -*- coding: utf-8 -*-

import jieba
import dfa

jieba.initialize()

# 载入自定义词典
jieba.load_userdict('./data/dict.txt')
# 不想分词的词
jieba.suggest_freq(('货物', '运输'), True)
jieba.suggest_freq(('车辆', '通行'), True)

# 敏感词
gfw = dfa.choose_filter("./data/bad.txt")

def multiple(num):
    jieba.enable_parallel(num)

def cut(text):
    return jieba.cut(text.replace(' ', ''))

def lcut(text):
    return "/".join(jieba.lcut(text.replace(' ', '')))

# 停用词
def stopwords():
    stopwords = set()
    with open('./data/stop.txt', 'r', encoding='utf-8', errors='ignore') as fp:
        for line in fp:
            stopwords.add(line.strip())
    return stopwords

def judge(text):
    text_doc_list = lcut(text)
    return gfw.judge(text_doc_list)

def filter(text):
    return gfw.filter(text)
