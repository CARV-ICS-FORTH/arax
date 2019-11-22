#!/bin/bash
exts="*.txt *.h *.c *.cpp"
echo -n 'Checking for line endings: '
fail=0
for ext in $exts
do
	for file in `find -name *"$ext"`
	do
		check=`file $file | grep CRLF`
		if [ "$check" != '' ]
		then
			echo
			echo -n "FIX ---> $file"
			fail=$((fail+1))
		fi
	done
done

if [ $fail -eq 0 ]
then
	echo "Ok"
else
	echo
	echo
	echo "$fail files with wrong line endings"
	exit 1
fi
