# Copyright(c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

FROM ubuntu:jammy

# 빌드 시 사용자 정보를 인자로 받기 (기본값을 현재 사용자로 설정)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# 패키지 설치
RUN apt update
RUN apt install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    git \
    make \
    cmake \
    g++ \
    libaio-dev \
    libgoogle-perftools-dev \
    libunwind-dev \
    clang-format \
    libboost-dev \
    libboost-program-options-dev \
    libmkl-full-dev \
    libcpprest-dev \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    sudo \
    vim \
    nano

# Python 심볼릭 링크 생성 (python 명령어로도 사용 가능)
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# pip 업그레이드 및 Python 패키지 설치
RUN python3 -m pip install --upgrade pip setuptools wheel

# 데이터 과학 라이브러리 설치
RUN python3 -m pip install \
    polars \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipython

# 사용자 그룹과 사용자 생성
RUN groupadd -g $GROUP_ID $USERNAME && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# PATH에 빌드 디렉토리 추가 (빌드 후 사용 가능)
ENV PATH="/workspace/build/apps:${PATH}"
ENV PATH="/workspace/build/apps/utils:${PATH}"

# 작업 디렉토리 생성 및 권한 설정
WORKDIR /workspace
RUN chown $USER_ID:$GROUP_ID /workspace

# 사용자 전환
USER $USERNAME

# 기본 명령어
CMD ["/bin/bash"]