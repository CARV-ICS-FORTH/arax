#!/bin/bash

#Concurrent builds
THREADS=8

#Run Tests
RUNTESTS=1

OPTIONS=`cat CMakeLists.txt | grep 'option(' |awk -F'(' '{print $2}'| awk '{print $1}'|uniq`
TARGETS="Debug" # Default"
SPINS="spin mutex ivshmem"
copts=""

status()
{
	echo -n -e '\e[1m' >> $4
	if [ $1 -eq 0 ]
	then
		echo -n -e '\e[38;5;082m['$2'] ' >> $4
	else
		echo -n -e '\e[38;5;198m['$3'] ' >> $4
	fi
	echo -n -e '\e[0m' >> $4
}

ptime()
{
	cat - | grep real|awk '{print $2}'|awk -Fs '{print $1}'|awk -Fm '{printf("%6.3f",$1*60+$2)}'
}

build()
{
	uid=$1
	shift
	home=$1
	shift
	build=$1
	shift
	buid=`basename $uid`
	log=$uid".log"
	blog=$uid".blog"
	cd $uid
	echo -n $buid | cut -d. -f 2 -z > $log
	echo $build | awk '{printf(" %-7s",$1);}' >> $log
	echo "$@" |sed 's/-D//g'|sed 's/=ON/=ON /g'|
	awk '{for(i=1;i<=NF;i++){split($i,v,"=");printf("%s ",v[2])}}'|
	awk -v lens="$lens" '{split(lens,l);for(i=1;i<=NF;i++){if($i=="ON")printf("%c[1;38;5;082m",27);if($i=="OFF")printf("%c[1;38;5;198m",27);printf("%*s ",l[i],$i)}printf("%c[0m",27);}'>> $log
	copts="-DCMAKE_BUILD_TYPE="$build" "$@
	(
		time cmake $copts ${home} &> $blog &&
		status $? PASS FAIL $log &&
		make &>> $blog
		bret=$?
		if [ $bret -eq 0 ]
		then
			echo >> "$home/"gm
		fi
		status $bret PASS FAIL $log
		warns=`grep 'warning:' $blog | awk 'END {printf("%04d",NR)}'`
		status $warns ZERO $warns $log
#		make test 2>&1| grep '%' | awk '{printf("%5.1f%%",$1)}' >> $log
	) 2>&1 | ptime >> $log
	echo >> $log
	if [ $bret -eq $warns ]
	then
		rm $blog
	fi
	cd - &> /dev/null
	rm -rf $uid

}

builds=0
pids=
files=
pids=
spawn_build()
{
	file=`mktemp -d`
	files[$builds]=$file".log"
	build $file `pwd` $@ &
	pids[$builds]=$!
	kill -STOP ${pids[builds]}
	builds=$((builds+1))
}

prod()
{
	local option=$1
	shift
	local old_copts=$copts" -D${option}=O"
	if [ $# -gt 0 ]
	then
		copts=${old_copts}"FF"
		prod $@
		copts=${old_copts}"N"
		prod $@
	else
		spawn_build $old_copts"FF"
		spawn_build $old_copts"N"
	fi
}
rm -Rf *log
cols="async_arch `echo $OPTIONS|awk '{for(i=1;i<=NF;i++)printf("%s ",$i)}'`"
lens=`echo $cols|awk '{for(i=1;i<=NF;i++)printf("%d ",length($i))}'`


for target in ${TARGETS}
do
	for spin in ${SPINS}
	do
		copts=$target" -Dasync_architecture="$spin
		prod $OPTIONS
	done
done
echo|awk -v cols="$cols" '{printf("%c[1mBuild ID   Target %s CMake  Make   Warn   Time%c[0m\n",27,cols,27);}'

jarr=
jcnt=0

run()
{
	for((b=0;b<$builds;b=b+1))
	do
		jarr[$jcnt]=${b}
		jcnt=$[jcnt+1]
		kill -18 ${pids[b]}
		if [ $jcnt == $THREADS ]
		then
			for (( i = 0 ; i < $THREADS ; i = i + 1 ))
			do
				wait ${pids[jarr[i]]}
				cat ${files[jarr[i]]}
				rm ${files[jarr[i]]}
			done
			jcnt=0
		fi
	done
	for (( i = 0 ; i < `echo $THREADS $builds | awk '{print $2%$1}'`; i = i + 1 ))
	do
		wait ${pids[jarr[i]]}
		cat ${files[jarr[i]]}
		rm ${files[jarr[i]]}
	done
	echo "Tested "$builds" configurations"
	echo "Builds "`wc -l gm| cut -d' ' -f 1`" succesfull"
	rm gm
}

time run
