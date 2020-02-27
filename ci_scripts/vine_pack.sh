# $1 Compress/Extract
# $2 Location
# $3 pack name
if [ $1 == 0 ]
then
	tar -czf $3.tgz -C $2 .
else
	mkdir -p $2
	tar -xzf $3.tgz -C $2
fi
