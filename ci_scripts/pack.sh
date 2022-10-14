# $1 Compress/Extract
# $2 Location
# $3 pack name
if [ $1 == 0 ]
then
	tar -czf /tmp/$3.tgz -C $2 .
	mv /tmp/$3.tgz $3.tgz
else
	mkdir -p $2
	tar -xzf $3.tgz -C $2
fi
