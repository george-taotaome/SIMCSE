# train
```
python train.py ./data/all.txt --epochs=10
````

# build index
```
python index.py --batch_size=256
```

# run service，[ip:8000/docs]: access swagger API
```
python main.py --workers=2
```

# build and run the Docker image to test
```
docker build -t simcse .
docker run -it --rm -p 8000:8000 --name simcse simcse
```

指标：暂时没空整理
相似度汇总取均值在这儿最大的用处是用来确定相似度多少才是合适的，并不是单纯用来衡量这个模型的准确度。
这个模型用于语义分析，喂的语料中有类似意思的文本，测试确保意思类似句子可以通过，语料中没有的句型测试不通过！
  首先确保不在语料中的文本无法通过，这是个硬指标！
  其次才是慢慢积累语料，让越来越多的文本喂养通过！
系统设计上还需持续做好语料收集、训练...