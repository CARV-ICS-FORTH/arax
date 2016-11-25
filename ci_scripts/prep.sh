cat /etc/pacman.d/mirrorlist | grep Greece -A 1 > mirrorlist
mv mirrorlist /etc/pacman.d/mirrorlist
yes | pacman -Suy
yes | pacman -S awk make cmake gcc htop tree check grep git
