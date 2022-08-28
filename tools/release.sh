#!/bin/bash
set -e

if [ -e ./build ]
then
	echo "Build directory exists, erase it"
	exit
fi

#Copy files and remove unwanted stuff
mkdir /tmp/release
cp -R * /tmp/release
rm -Rf /tmp/release/ci_scripts
rm -Rf /tmp/release/build/*
rm -Rf /tmp/release/.git
# Add License notice on all c h cpp files
curl http://www.apache.org/licenses/LICENSE-2.0.txt > /tmp/release/LICENSE

c_files=`find /tmp/release -name '*.c' | grep -v 3rdparty`
cpp_files=`find /tmp/release -name '*.cpp' | grep -v 3rdparty`
head_files=`find /tmp/release -name '*.h' | grep -v 3rdparty`

for file in $c_files  $cpp_files $head_files
do
	cat ./tools/lic $file > $file.lc
	mv $file.lc $file
done

cd /tmp/release
#./tools/build_check.sh
rm -Rf /tmp/release/tools
sed -i '/carvgit.ics.forth.gr/d' README.md
sed -i '/pre-commit/d' CMakeLists.txt
sed -i '/hooks/d' CMakeLists.txt
rm -Rf /tmp/release/cmake/pre-commit
cd ..

git clone git@github.com:vineyard2020/vine_talk.git
cp -R release/* ./vine_talk
cd ./vine_talk
git add src
git add tests
git add misc
git add cmake
git add noop
git add JVTalk
git add vdf
git add araxgrind
git add include
git add README.md
git add CMakeLists.txt
