
FROM python:3.10-slim

WORKDIR /app


COPY . /app


RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 unzip && rm -rf /var/lib/apt/lists/*


RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt


EXPOSE 8080

CMD ["python3", "app.py"]