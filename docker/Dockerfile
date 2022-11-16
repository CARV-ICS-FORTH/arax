# Transport Images
FROM DOCK_HOST/accelerators/cuda:10.1-base-ubuntu18.04 as arax-cmake
ENV JOBS=4
RUN apt-get -qqy update
RUN apt-get -qqy install git g++ make wget libssl-dev pkg-config libncursesw5-dev libncurses-dev
RUN wget "https://github.com/Kitware/CMake/releases/download/v3.22.3/cmake-3.22.3-linux-x86_64.sh"
RUN bash ./cmake-3.22.3-linux-x86_64.sh --skip-license
RUN rm -Rf cmake-3.22.3-linux-x86_64.sh /doc /man /share/cmake-3.22/Help /bin/cmake-gui

FROM arax-cmake as arax-build-deps
COPY . arax
WORKDIR /arax/build
RUN cmake -DBUILD_TESTS=OFF -DARAX_OBJECT_NAME_SIZE=128 ..

FROM arax-build-deps as arax-base-build
RUN make -j ${JOBS}
RUN make install

FROM scratch as arax-base
COPY --from=arax-base-build / /
RUN apt-get -qqy update
RUN apt-get -qqy install pkg-config git
RUN apt-get remove wget -qqy
RUN rm -Rf /arax
RUN rm -Rf /var/cache/*
RUN rm -Rf /var/log/*
RUN rm -rf /var/lib/apt/lists/*

FROM arax-build-deps as arax-install-build
RUN make -j ${JOBS}
RUN make DESTDIR=/install install

FROM scratch as arax-install
#@
COPY --from=arax-install-build /install /

FROM DOCK_HOST/accelerators/cuda:10.1-base-ubuntu18.04 as arax-devel
#@
COPY --from=arax-base / /
RUN apt-get clean
RUN rm -Rf /var/cache/*
RUN rm -Rf /var/log/*
RUN rm -rf /var/lib/apt/lists/*

FROM DOCK_HOST/accelerators/cuda:10.1-base-ubuntu18.04 as arax-runtime
#@
COPY --from=arax-install / /

# Controller Images

FROM DOCK_HOST/accelerators/cuda:10.1-cudnn7-devel-ubuntu18.04 as arax-controller-base
COPY --from=arax-devel / /

# Cuda Hacks - Not proud, but necessary
RUN ln -s /usr/local/cuda-10.1/lib64/libOpenCL.so /usr/lib/libOpenCL.so
RUN ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so /usr/lib/x86_64-linux-gnu/libcudnn.so.7.0
RUN ln /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusparse.so /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusparse.so.10.1
RUN ln /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusolver.so /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcusolver.so.10.1
RUN ln /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcufft.so /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcufft.so.10.1
RUN ln /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so /usr/local/cuda-10.1/targets/x86_64-linux/lib/libcurand.so.10.1
RUN ln /usr/lib/x86_64-linux-gnu/libcublas.so /usr/lib/x86_64-linux-gnu/libcublas.so.10.1

FROM arax-controller-base as arax-controller-base-build
COPY . arax-controller
WORKDIR arax-controller/build
RUN apt-get -qqy update 
RUN apt-get -qqy install pkg-config git
RUN cmake -DENABLE_Cpu=ON -DENABLE_Cuda=ON -DENABLE_OpenCL=ON -DBUILTINS_PATH='/usr/lib/araxcontroller' -DARAX_OBJECT_NAME_SIZE=128 ..
RUN make -j ${JOBS}
RUN make DESTDIR=/install install
COPY ../controller/docker/arax_start.sh /install/usr/bin/arax_start.sh
COPY ../controller/docker/arax_stop.sh /install/usr/bin/arax_stop.sh

FROM scratch as araxcontroller-install
#@
COPY --from=arax-controller-base-build /install /

FROM arax-controller-base as araxcontroller-devel
#@
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.1/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu
COPY --from=araxcontroller-install / /
COPY ../controller/docker/entry.sh /bin/entry
ENTRYPOINT ["/bin/bash","/bin/entry"]

FROM DOCK_HOST/accelerators/cuda:10.1-cudnn7-runtime-ubuntu18.04 as araxcontroller-runtime
#@
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.1/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu
COPY --from=arax-install / /
COPY --from=araxcontroller-install / /
COPY ../controller/docker/entry.sh /bin/entry
ENTRYPOINT ["/bin/bash","/bin/entry"]
