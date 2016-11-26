if [ $1 == 0 ]
then
	tar -cf packages.tar /var/cache/pacman/pkg/
else
	tar -xf -C /var/cache/pacman/pkg/ packages.tar 
fi
