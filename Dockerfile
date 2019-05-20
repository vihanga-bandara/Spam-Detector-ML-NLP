FROM ubuntu:18.04

MAINTAINER Vihanga Bandara "vihanga.2015072@iit.ac.lk"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 5000

ENTRYPOINT ["python3"]

CMD [ "application.py" ]