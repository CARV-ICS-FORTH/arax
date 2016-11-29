mkdir build
cd build
cmake ..
make -j install
cd ..
wget https://carvgit.ics.forth.gr/vineyard/vine_controller/repository/archive.tar?ref=master
tar xvf *.tar
rm *.tar
cd vine_controller*
mkdir build
cd build
cmake ..
make
