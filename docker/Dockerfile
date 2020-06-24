# From https://github.com/ufoym/deepo/blob/master/docker/Dockerfile.pytorch-py36-cu90

# ==================================================================
# module list
# ------------------------------------------------------------------
# python        3.6    (apt)
# pytorch       1.2.0  (pip)
# ==================================================================

FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
ENV CONDA_DIR /opt/conda
ENV PATH ${CONDA_DIR}/bin:$PATH
RUN mkdir -p ${CONDA_DIR} && \
    echo export PATH=${CONDA_DIR}/bin:'$PATH' > /etc/profile.d/conda.sh
RUN APT_INSTALL="apt-get install -y --no-install-recommends" && \
    rm -rf /var/lib/apt/lists/* \
           /etc/apt/sources.list.d/cuda.list \
           /etc/apt/sources.list.d/nvidia-ml.list && \
    apt-get update && \
# ==================================================================
# tools
# ------------------------------------------------------------------
    DEBIAN_FRONTEND=noninteractive $APT_INSTALL \
        build-essential \
        ca-certificates \
        wget \
        git \
        vim \
        fish \
        libsparsehash-dev \
        zip \
        unzip \
        tar \
        sudo \
        curl \
        rsync \
        gnuplot-x11 \
        ghostscript \
        texlive-extra-utils \
        apt-utils \
        gcc \
        g++ \
        graphviz \
        libprotobuf-dev \
        protobuf-compiler \
        software-properties-common \
        python-software-properties \
        python3-dev \
        python3-pip \
        python3-tk \
        python3-dbg \
        python3-wheel \
        python3-setuptools \
        libhdf5-dev \
        libopenblas-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        locales
# ==================================================================
# python
# ------------------------------------------------------------------
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    /bin/bash /Miniconda3-latest-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-latest-Linux-x86_64.sh
ARG python_version=3.6.7
RUN conda install -y python=${python_version}
RUN pip install -U pip
RUN conda install \
        cudatoolkit==10.2.89 \
        h5py==2.9.0 \
        matplotlib==3.1.1 \
        mkl==2019.0 \
        nose==1.3.7 \
        notebook==6.0.2 \
        pandas==0.25.3 \
        pillow==6.2.1 \
        pyyaml==5.2 \
        scikit-image==0.15.0 \
        scikit-learn==0.22.1 \
        scipy==1.3.2 \
        six==1.13.0
RUN pip install \
        addict==2.2.1 \
        cachetools==3.1.0 \
        cython==0.29.14 \
        descartes==1.1.0 \
        easydict==1.9 \
        fire==0.2.1 \
        future==0.18.2 \
        imagecorruptions==1.1.0 \
        joblib==0.14.1 \
        jupyter==1.0.0 \
        llvmlite==0.31.0 \
        motmetrics==1.1.3 \
        numpy==1.18.1 \
        numba==0.47 \
        objgraph==3.4.1 \
        open3d-python==0.7.0.0 \
        opencv-python==4.1.2.30 \
        opencv-contrib-python==4.1.2.30 \
        optuna==1.4.0 \
        protobuf==3.11.2 \
        psutil==5.6.7 \
        pybind11==2.4.3 \
        pyquaternion==0.9.5 \
        pytest-runner==5.2 \
        requests==2.24.0 \
        seaborn==0.9.0 \
        shapely==1.6.4.post2 \
        terminaltables==3.1.0 \
        tqdm==4.46.0 \
        vtk==8.1.2
RUN pip install \
        lyft-dataset-sdk==0.0.8 \
        pycocotools==2.0.0
# ==================================================================
# pytorch & tensorflow
# ------------------------------------------------------------------
RUN pip install \
        tensorflow==1.14.0 \
        tensorboardX==2.0
RUN conda install \
        pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
# ==================================================================
# config & cleanup
# ------------------------------------------------------------------
RUN ldconfig && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* ~/*
# ==================================================================
# spconv
# use a commit 73427720a539caf9a44ec58abe3af7aa9ddb8e39 because Det3D only supports spconv1.0
# ------------------------------------------------------------------
WORKDIR /opt
RUN wget https://dl.bintray.com/boostorg/release/1.68.0/source/boost_1_68_0.tar.gz
RUN tar xzvf boost_1_68_0.tar.gz
RUN cp -r ./boost_1_68_0/boost /usr/include
RUN rm -rf ./boost_1_68_0
RUN rm -rf ./boost_1_68_0.tar.gz
RUN wget https://github.com/Kitware/CMake/releases/download/v3.14.4/cmake-3.14.4.tar.gz
RUN tar zxvf cmake-3.14.4.tar.gz
RUN cd ./cmake-3.14.4 && ./configure && make && make install && cd ..
RUN rm -rf ./cmake-3.14.4 && rm -rf ./cmake-3.14.4.tar.gz
RUN git clone --depth 99 --recursive https://github.com/traveller59/spconv.git
RUN cd ./spconv && \
    git checkout 73427720a539caf9a44ec58abe3af7aa9ddb8e39 && \
    python setup.py bdist_wheel && \
    pip install /opt/spconv/dist/spconv-1.0-cp36-cp36m-linux_x86_64.whl && \
    cd ..
# ==================================================================
# apex for training with FP16 mixed precision
# use a commit b5a7c5f972bc644804408dd96d7cb515df3af07b because of https://github.com/NVIDIA/apex/issues/690
# ------------------------------------------------------------------
RUN git clone --depth 99 https://github.com/NVIDIA/apex
RUN cd ./apex && \
    git checkout b5a7c5f972bc644804408dd96d7cb515df3af07b && \
    pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" . && cd ..
# ==================================================================
# nuScenes devkit
# ------------------------------------------------------------------
RUN git clone https://github.com/tianweiy/nuscenes-devkit
# ==================================================================
# numba
# ------------------------------------------------------------------
ENV NUMBA_CUDA_DRIVER=/usr/lib/x86_64-linux-gnu/libcuda.so
# ==================================================================
# others
# ------------------------------------------------------------------
# Set the locale
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja:en
ENV LC_ALL ja_JP.UTF-8
# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

# User env
ARG USERNAME=docker
ARG UID=1000
ARG GID=1000
RUN useradd -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod --uid $UID $USERNAME && \
    groupmod --gid $GID $USERNAME
