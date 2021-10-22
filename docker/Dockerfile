FROM harbor.port0.org/dhub/nvidia/cuda:10.1-base-ubuntu18.04 as vinetalk-cmake
ENV JOBS=4
RUN apt-get -qqy update
RUN apt-get -qqy install git g++ make wget libssl-dev pkg-config libncurses-dev
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-x86_64.sh"
RUN bash ./cmake-3.22.3-linux-x86_64.sh --skip-license
RUN rm -Rf cmake-3.22.3-linux-x86_64.sh /doc /man /share/cmake-3.22/Help

FROM vinetalk-cmake as vinetalk-build-deps
COPY . vine_talk
WORKDIR /vine_talk/build
RUN cmake -DBUILD_TESTS=OFF -DVINE_OBJECT_NAME_SIZE=128 ..

FROM vinetalk-build-deps as vinetalk-base-build
RUN make -j ${JOBS}
RUN make install

FROM scratch as vinetalk-base
COPY --from=vinetalk-base-build / /
RUN apt-get remove perl wget -qqy
RUN rm -Rf /vine_talk

FROM vinetalk-build-deps as vinetalk-install-build
RUN make -j ${JOBS}
RUN make DESTDIR=/install install

FROM scratch as vinetalk-install
COPY --from=vinetalk-install-build /install /

FROM harbor.port0.org/dhub/nvidia/cuda:10.1-base-ubuntu18.04 as vinetalk-devel
RUN apt-get -qqy update
RUN apt-get -qqy remove perl
RUN apt-get -qqy install pkg-config
COPY --from=vinetalk-base / /
RUN apt-get clean
RUN rm -Rf /var/cache/*
RUN rm -Rf /var/log/*
RUN rm -rf /var/lib/apt/lists/*

FROM harbor.port0.org/dhub/nvidia/cuda:10.1-base-ubuntu18.04 as vinetalk-runtime
COPY --from=vinetalk-install / /
