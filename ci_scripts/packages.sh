if [ $1 == 0 ]
then
#	du -h /var/cache/pacman/pkg/
#	tar -cf packages.tar /var/cache/pacman/pkg/
else
#	tar -xf packages.tar  -C /
	pwd
	bash prep.sh
fi
