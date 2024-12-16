FROM python:3.10
RUN apt update && \
    apt install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt Makefile ./
RUN make init

COPY . .

CMD ["make", "run"]
 
