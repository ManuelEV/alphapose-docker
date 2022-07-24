FROM nvidia/cuda:11.7.0-runtime-ubuntu20.04
# FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04

# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl

ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && chmod +x ~/miniconda.sh \
    && ~/miniconda.sh -b -p /opt/miniconda \
    && rm ~/miniconda.sh \
    && sed -i "$ a PATH=/opt/miniconda/bin:\$PATH" /etc/environment

# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

# Alphapose dependencies
RUN conda create -n alphapose python=3.6 -y
RUN conda install pytorch torchvision cudatoolkit=11.3 -c pytorch -y
RUN export PATH=/usr/local/cuda/bin/:$PATH
RUN export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
RUN python -m pip install cython
RUN apt-get install libevent-dev -y
RUN apt-get install gcc python-dev libjpeg-dev libfreetype6 libfreetype6-dev zlib1g-dev -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install cython_bbox

# Copy app files
COPY ./lab-01 /lab-01
WORKDIR /lab-01/

COPY ./requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m pip install opencv-python
RUN python -m pip install gunicorn

EXPOSE 5000

# CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "wsgi:app" ]
CMD [ "tail", "-F", "anything" ]
# CMD ["gunicorn", "-w 4", "-b", "0.0.0.0:5000", "wsgi:app"]


# gunicorn --bind 0.0.0.0:5000 wsgi:app

# python -m pip install
# python -m flask run
# sudo docker run -d -p 5000:5000 --gpus all --name alphapose  cuda-image
# sudo docker build . -t cuda-image