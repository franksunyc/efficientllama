FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# prepare for install
RUN apt update && \
    apt -y upgrade

# install pip
RUN apt install -y python3-pip git

# install requirements
COPY requirements.txt ./
RUN pip3 install --no-cache-dir --upgrade -r requirements.txt
RUN rm ./requirements.txt
