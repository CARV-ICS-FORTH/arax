if [ $1 == 0 ]
then
	tar -cf packages.tar /var/cache/pacman/pkg/
else
	tar -xf packages.tar  -C /var/cache/pacman/pkg/
fi
