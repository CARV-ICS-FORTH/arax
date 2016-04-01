libs=`ls ../libvine_talk_*.a`
versions=${libs/"../libvine_talk_"/}
versions=${versions/".a"/}

FLAGS='-g -lrt -I ../'
echo $versions
for file in `ls *.c`
do
	for version in "${versions}"
	do
#		clear
		echo "Building \"${file/'.c'/}.$version\""
		echo gcc $file $FLAGS ../libvine_talk_$version.a -o ${file/'.c'/}.$version
	done
done
