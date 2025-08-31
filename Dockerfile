# Copyright(c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT license.

FROM ubuntu:jammy

# 빌드 시 사용자 정보를 인자로 받기 (기본값을 현재 사용자로 설정)
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG USERNAME=user

# 패키지 설치
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:git-core/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
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
    sudo \
    vim \
    nano \
    wget \
    libomp5 \
    && rm -rf /var/lib/apt/lists/*
    # --- libomp5 추가 ---

# Python 심볼릭 링크 생성
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# --- START: Conda 및 RAPIDS(cuML) 설치 ---

# 1. Conda 설치
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py310_24.4.0-0-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tip && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# 2. Conda 환경 PATH 설정
ENV PATH=/opt/conda/bin:$PATH

# 3. Conda 환경 생성 및 모든 파이썬 라이브러리 설치
#    pip 대신 conda로 통합하여 의존성을 안정적으로 관리합니다.
RUN conda create -n rapids -c rapidsai -c conda-forge -c nvidia \
    cuml \
    cupy \
    pandas \
    numpy \
    scikit-learn \
    matplotlib \
    seaborn \
    jupyter \
    ipython \
    tqdm \
    polars \
    python=3.10 -y

# --- END: Conda 및 RAPIDS(cuML) 설치 ---

# 사용자 그룹과 사용자 생성
RUN groupadd -g $GROUP_ID $USERNAME && \
    useradd -u $USER_ID -g $GROUP_ID -m -s /bin/bash $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# PATH에 빌드 디렉토리 추가 (빌드 후 사용 가능)
ENV PATH="/workspace/build/apps:${PATH}"
ENV PATH="/workspace/build/apps/utils:${PATH}"

# 작업 디렉토리 생성 및 권한 설정
WORKDIR /workspace
RUN chown -R $USER_ID:$GROUP_ID /workspace

# 사용자 전환
USER $USERNAME

# 컨테이너 시작 시 Conda 환경 자동 활성화
SHELL ["/bin/bash", "-c"]
ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "rapids", "--no-capture-output"]
CMD ["/bin/bash"]