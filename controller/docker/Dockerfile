FROM harbor.port0.org/accelerators/arax-install:ARAX_BRANCH as vt-install

FROM harbor.port0.org/accelerators/arax-devel:ARAX_BRANCH as vt-devel

FROM harbor.port0.org/accelerators/cuda:10.1-cudnn7-devel-ubuntu18.04 as cuda-devel

FROM harbor.port0.org/accelerators/cuda:10.1-cudnn7-runtime-ubuntu18.04 as cuda-runtime

FROM scratch as arax-controller-base
COPY --from=vt-devel / /
COPY --from=cuda-devel / /

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
RUN cmake -DENABLE_Cpu=ON -DENABLE_Cuda=ON -DENABLE_OpenCL=ON -DBUILTINS_PATH='/usr/lib/araxcontroller' ..
RUN make -j ${JOBS}
RUN make DESTDIR=/install install
COPY ../docker/arax_start.sh /install/usr/bin/arax_start.sh
COPY ../docker/arax_stop.sh /install/usr/bin/arax_stop.sh

FROM scratch as araxcontroller-install
COPY --from=arax-controller-base-build /install /

FROM arax-controller-base as araxcontroller-devel
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.1/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu
COPY --from=araxcontroller-install / /
COPY docker/entry.sh /bin/entry
ENTRYPOINT ["/bin/bash","/bin/entry"]

FROM cuda-runtime as araxcontroller-runtime
ENV LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib/:/usr/local/cuda-10.1/extras/CUPTI/lib64:/usr/lib/x86_64-linux-gnu
COPY --from=vt-install / /
COPY --from=araxcontroller-install / /
COPY docker/entry.sh /bin/entry
ENTRYPOINT ["/bin/bash","/bin/entry"]
