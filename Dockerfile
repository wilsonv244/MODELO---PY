FROM python:latest
WORKDIR /Modelo Grupo Solidario
COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends \
    unixodbc-dev \
    unixodbc \
    libpq-dev 
RUN pip3 install -U scikit-learn scipy matplotlib
RUN pip3 install pydotplus
RUN pip3 install -r requirements.txt
COPY . .
