if [ $1 == 0 ]
then
	true
#	du -h /var/cache/pacman/pkg/
#	tar -cf packages.tar /var/cache/pacman/pkg/
else
#	tar -xf packages.tar  -C /
	pwd
# Greece Servers FTW
echo 'Server = http://ftp.cc.uoc.gr/mirrors/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://ftp.ntua.gr/pub/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://foss.aueb.gr/mirrors/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://ftp.otenet.gr/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
yes | pacman -Syy
yes | pacman -S archlinux-keyring
yes | pacman -Suy
yes | pacman -S awk make cmake gcc htop tree check git grep wget

fi
