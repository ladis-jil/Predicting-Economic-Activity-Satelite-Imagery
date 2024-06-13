FROM python:3.10.6-buster

COPY satellite /satellite
COPY requirements_pred.txt /requirements.txt

RUN pip install --upgrade pip
RUN apt-get update && apt-get install -y libhdf5-dev
RUN pip install --no-binary h5py h5py
RUN pip install -r requirements.txt

CMD uvicorn satellite.api.lib:app --host 0.0.0.0 --port $PORT
