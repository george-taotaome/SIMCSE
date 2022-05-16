# SIMCSE
## train
```
python train.py ./data/all.txt --epochs=10
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
