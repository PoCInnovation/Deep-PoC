FROM python:3.8-slim-buster 

WORKDIR /app
COPY . . 
RUN pip3 install -r ./requirements.txt
EXPOSE 8000

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python3.8", "manage.py", "runserver", "0.0.0.0:8000"]