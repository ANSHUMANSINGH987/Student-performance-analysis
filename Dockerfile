FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 unzip -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["python3", "app.py"]