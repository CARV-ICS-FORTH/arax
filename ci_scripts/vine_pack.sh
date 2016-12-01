# $1 Compress/Extract
# $2 Location
# $3 pack name
if [ $1 == 0 ]
then
	tar -cf $3.tar -C $2 .
else
	tar -xf $3.tar  -C $2
	tree $2
fi
