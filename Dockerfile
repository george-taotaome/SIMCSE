FROM python:3.7

WORKDIR /usr/src/app

RUN sed -i -E 's/(deb|security).debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && \
    apt-get install -y libopenblas-dev libomp-dev && \
    ln -s /usr/lib/llvm-11/lib/libomp.so /usr/lib && \
    ldconfig && \
    apt-get clean -y && \
    apt-get autoclean -y && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* /var/lib/log/* /tmp/* /var/tmp/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple/

COPY ./src .

EXPOSE 8000
CMD ["python", "main.py"]%
