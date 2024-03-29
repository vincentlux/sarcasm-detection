# FROM python:3.6-alpine
# https://github.com/anibali/docker-pytorch/blob/master/cuda-9.0/Dockerfile
FROM nvidia/cuda:9.0-base-ubuntu16.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda
RUN curl -so ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh
ENV PATH=/home/user/miniconda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Create a Python 3.6 environment
RUN /home/user/miniconda/bin/conda install conda-build \
 && /home/user/miniconda/bin/conda create -y --name py36 python=3.6.5 \
 && /home/user/miniconda/bin/conda clean -ya
ENV CONDA_DEFAULT_ENV=py36
ENV CONDA_PREFIX=/home/user/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH

# CUDA 9.0-specific steps
RUN conda install -y -c pytorch \
    cuda90=1.0 \
    magma-cuda90=2.3.0 \
    "pytorch=0.4.1=py36_cuda9.0.176_cudnn7.1.2_1" \
    torchvision=0.2.1 \
 && conda clean -ya


# Install Requests, a Python library for making HTTP requests
# RUN conda install -y requests=2.19.1 \
#  && conda clean -ya

# Install Graphviz
# RUN conda install -y graphviz=2.38.0 \
#  && conda clean -ya
# RUN pip install graphviz==0.8.4

# Install OpenCV3 Python bindings
# RUN sudo apt-get update && sudo apt-get install -y --no-install-recommends \
#     libgtk2.0-0 \
#     libcanberra-gtk-module \
#  && sudo rm -rf /var/lib/apt/lists/*
# RUN conda install -y -c menpo opencv3=3.1.0 \
#  && conda clean -ya

## Spacy
RUN pip install spacy &&\
    python -m spacy download en


COPY ./requirements.txt .
RUN pip install -r requirements.txt
# Set the default command to python3
CMD /bin/bash
# CMD ["python -m pip install -U spacy && python -m spacy download en"]
