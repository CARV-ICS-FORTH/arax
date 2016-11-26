if [ $1 == 0 ]
then
	tar -cf packages.tar /var/cache/pacman/pkg/
else
	tar -xvf packages.tar  -C /
fi
