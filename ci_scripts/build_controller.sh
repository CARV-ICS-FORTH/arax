mkdir build
cd build
cmake ..
make -j install
cd ..
wget https://carvgit.ics.forth.gr/accelerators/arax_controller/repository/archive.tar?ref=master
tar xvf *.tar
rm *.tar
cd arax_controller*
mkdir build
cd build
cmake ..
make
