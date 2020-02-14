if [ $1 == 0 ]
then
	df -h /var/cache/pacman/pkg/
	tar -cf packages.tar /var/cache/pacman/pkg/
else
	tar -xf packages.tar  -C /
fi
