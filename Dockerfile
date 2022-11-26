FROM python:3.9.2

ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update -y && \
    apt-get install build-essential libssl-dev cmake -y && \
    apt-get install ffmpeg libsm6 libxext6  -y

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

COPY . ./

ENV PYTHONPATH app

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
