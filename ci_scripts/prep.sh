yes | pacman -Suy
yes | pacman -S grep
cat /etc/pacman.d/mirrorlist
cat /etc/pacman.d/mirrorlist | grep Greece -A 1 > mirrorlist
echo "AFTER"
cat /etc/pacman.d/mirrorlist
mv mirrorlist /etc/pacman.d/mirrorlist
yes | pacman -S awk make cmake gcc htop tree check git
