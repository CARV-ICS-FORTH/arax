if [ $1 == 0 ]
then
	tar -cf packages.tar /var/cache/pacman/pkg/
else
	tar  -C /var/cache/pacman/pkg/ xf packages.tar 
fi
