FROM harbor.port0.org/accelerators/cuda:10.1-base-ubuntu18.04 as arax-cmake
ENV JOBS=4
RUN apt-get -qqy update
RUN apt-get -qqy install git g++ make wget libssl-dev pkg-config libncursesw5-dev libncurses-dev
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-x86_64.sh"
RUN bash ./cmake-3.22.3-linux-x86_64.sh --skip-license
RUN rm -Rf cmake-3.22.3-linux-x86_64.sh /doc /man /share/cmake-3.22/Help

FROM arax-cmake as arax-build-deps
COPY . arax
WORKDIR /arax/build
RUN cmake -DBUILD_TESTS=OFF -DARAX_OBJECT_NAME_SIZE=128 ..

FROM arax-build-deps as arax-base-build
RUN make -j ${JOBS}
RUN make install

FROM scratch as arax-base
COPY --from=arax-base-build / /
RUN apt-get remove perl wget -qqy
RUN rm -Rf /arax

FROM arax-build-deps as arax-install-build
RUN make -j ${JOBS}
RUN make DESTDIR=/install install

FROM scratch as arax-install
COPY --from=arax-install-build /install /

FROM harbor.port0.org/accelerators/cuda:10.1-base-ubuntu18.04 as arax-devel
RUN apt-get -qqy update
RUN apt-get -qqy remove perl
RUN apt-get -qqy install pkg-config
COPY --from=arax-base / /
RUN apt-get clean
RUN rm -Rf /var/cache/*
RUN rm -Rf /var/log/*
RUN rm -rf /var/lib/apt/lists/*

FROM harbor.port0.org/accelerators/cuda:10.1-base-ubuntu18.04 as arax-runtime
COPY --from=arax-install / /
