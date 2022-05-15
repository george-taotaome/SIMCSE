# SIMCSE
## train
```
python train.py ./data/all.txt --epochs=8
````

## build index
```
python index.py
```

## run service，[ip:8000/docs]: access swagger API
```
python main.py --workers=2
```

## build and run the Docker image to test
```
docker build -t simcse .
docker run -it --rm -p 8000:8000 --name simcse simcse
```
---
系统设计上还需持续做好语料收集、训练..  
相似度汇总取均值在这儿最大的用处是用来确定相似度多少才是合适的，并不是单纯用来衡量这个模型的准确度。  
这个模型用于语义分析，喂的语料中有类似意思的文本，测试确保意思类似句子可以通过，语料中没有的句型测试不通过！
* 首先确保不在语料中的文本无法通过，这是个硬指标！
* 其次才是慢慢积累语料，让越来越多的文本喂养通过！  
---
如须验证，可将数据集划分下： 
*  随机抽取80% 放到训练集中
* 在剩下的20%中随机抽取10%到测试集中，每条对应再从上面训练集中抽取一条训练数据，然后人工打标是否匹配
* 剩下的作为验证集，类似上面也是每条数据也抽取训练集中一条数据对应，人工打标是否匹配  

---
* 过拟合
* 欠拟合

---
测试1： 使用 model = epoch_16-batch_6000-loss_0.029720
正常样本：最小相似度=0， 52条数据<0.59，52/999=0.052，大于0.59相似度的合格率为94.8%  具体看positive_old.csv
异常样本：最大相似度=0.58946418762207  具体看negative_old.csv

测试2： 使用 model = epoch_16-batch_6000-loss_0.029720，新增85条新样本重新训练
正常样本：最小相似度=0.9047877788543700，大于0.59相似度的合格率为100%  具体看positive_old.csv
异常样本：最大相似度=0.587344437837601  具体看negative_old.csv