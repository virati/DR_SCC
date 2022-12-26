FROM python:3.10

WORKDIR /jup

COPY requirements.txt .
RUN pip install --upgrade pip && pip install jupyter -U && pip install jupyterlab && pip install -r requirements.txt

RUN mkdir -p /home/notebooks
WORKDIR /home/notebooks

COPY notebooks/ notebooks/

RUN mkdir /home/hume/analysis_in -p

EXPOSE 8888

WORKDIR /home/notebooks/

ENTRYPOINT ["jupyter", "lab","--port=8888","--ip=0.0.0.0","--allow-root","--NotebookApp.token=''","--NotebookApp.password=''"]