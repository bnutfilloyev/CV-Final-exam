FROM python:3.9.2

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update -y && \
    apt-get install build-essential libssl-dev cmake -y && \
    apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/requirements.txt

RUN pip install -r requirements.txt

COPY . ./

ENV PYTHONPATH app

ENTRYPOINT ["python", "train.py"]