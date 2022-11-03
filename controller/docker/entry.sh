# Have to do this at runtime
ln -s /usr/lib/x86_64-linux-gnu/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.10.1
exec "$@"
