
FROM pytorch/pytorch


SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    apt-get purge --autoremove -y curl \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        apt-utils \
        build-essential \
        curl \
        xvfb \
        ffmpeg \
        xorg-dev \
        libsdl2-dev \
        swig \
        cmake \
        python-opengl \
        tmux \
        wget \
        unar \
        unrar \
        unzip

RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6

RUN pip3 install --upgrade pip

RUN pip3 install numpy \
                 gym \
                 pyglet \
                 box2d-py \
                 matplotlib \
                 seaborn \
                 pandas \
                 notebook \
                 scikit-image \
                 atari_py \
                 opencv-python

RUN wget http://www.atarimania.com/roms/Roms.rar
RUN mkdir extracted_roms && mv Roms.rar extracted_roms && cd extracted_roms && unrar e -y Roms.rar
RUN python3 -m atari_py.import_roms extracted_roms
RUN pip install gym[atari,accept-rom-license]==0.21.0

RUN apt-get install -y vim

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute
