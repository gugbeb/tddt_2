# syntax=docker/dockerfile:1
FROM flatironinstitute/triqs:3.3.1 AS base
LABEL maintainer="Igor Krivenko"
LABEL description="Implementation of the time-dependent dual TRILEX theory"
LABEL version="0.3.3"
LABEL org.opencontainers.image.source=https://github.com/krivenko/tddt

USER root
RUN useradd -m -s /bin/bash -u 999 build && echo "build:build" | chpasswd
RUN usermod -aG sudo build
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make g++ file apt-utils vim vim-python-jedi python3-venv

ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

USER build
RUN mkdir /home/build/realevol
WORKDIR /home/build/realevol

# Install ARPACK-NG
RUN git clone https://github.com/opencollab/arpack-ng.git arpack-ng.git
RUN mkdir arpack-ng.build
WORKDIR arpack-ng.build
RUN cmake ../arpack-ng.git                              \
        -DCMAKE_INSTALL_PREFIX=/usr                     \
        -DCMAKE_BUILD_TYPE=Release                      \
        -DBUILD_SHARED_LIBS=ON                          \
        -DICB=ON                                        \
        -DMPI=ON
RUN make -j6 && ctest --output-on-failure
USER root
RUN make install

# Install realevol
USER build
WORKDIR /home/build/realevol

RUN git clone https://github.com/krivenko/triqs-realevol.git realevol.git
RUN mkdir realevol.build
WORKDIR realevol.build
ENV CC=gcc CXX=g++ REPO=/build/repo
RUN cmake ../realevol.git                               \
        -DCMAKE_INSTALL_PREFIX=/usr                     \
        -DCMAKE_BUILD_TYPE=Release                      \
        -Darpack-ng_DIR=/usr/lib/x86_64-linux-gnu/cmake \
        -DBUILD_SHARED_LIBS=ON                          \
        -DBuild_Tests=ON                                \
        -DBuild_Benchmarks=OFF                          \
        -DBUILD_DEBIAN_PACKAGE=ON
RUN make -j6 VERBOSE=1 && ctest --output-on-failure && cpack
USER root
RUN make install && mkdir -p $REPO && mv *.deb $REPO

# Install TDDT
USER root
COPY --chown=root . /src/tddt
WORKDIR /src/tddt
RUN pip3 install --user --break-system-packages -r requirements.txt
RUN pip3 install --user --break-system-packages .

# Test TDDT
RUN python3 -m pytest -v --with-mpi

# Cleanup build files
USER root
RUN rm -rf /home/build/arpack-ng.* /home/build/realevol.*
