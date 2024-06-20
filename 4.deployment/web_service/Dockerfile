FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip

WORKDIR /app

RUN mkdir -p output/yellow

COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "--timeout=300", "predict:app" ]
