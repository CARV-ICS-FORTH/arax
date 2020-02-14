# Greece Servers FTW
echo 'Server = http://ftp.cc.uoc.gr/mirrors/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://ftp.ntua.gr/pub/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://foss.aueb.gr/mirrors/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
echo 'Server = http://ftp.otenet.gr/linux/archlinux/$repo/os/$arch' > /etc/pacman.d/mirrorlist
yes | pacman -S archlinux-keyring
yes | pacman -Suy
yes | pacman -S awk make cmake gcc htop tree check git grep wget
