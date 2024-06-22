FROM python:3.10.9
COPY . .

WORKDIR /

RUN mkdir -p /.cache/huggingface/hub

# Separate RUN command for chmod
RUN chmod -R 777 /.cache/huggingface

RUN pip install --no-cache-dir --upgrade -r /requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
