#!/usr/bin/python3
import subprocess

# Get list of basic options
basic = subprocess.run(["cmake","..","-LH"],capture_output=True,encoding='UTF8').stdout.replace('\n\n','\n').split('\n')
# Get list of all (basic+advanced) options
advanced = subprocess.run(["cmake","..","-LAH"],capture_output=True,encoding='UTF8').stdout.replace('\n\n','\n').split('\n')

# Remove lines starting with -- (i.e. cmake messages)
basic = [line for line in basic if not line.startswith('--')]
advanced = [line for line in advanced if not line.startswith('--')]

def lines2map(lines):
	ret_map = {}
	for i in range(0,len(lines)-1,2):
		option = lines[i+1]
		if option.startswith("Poco_"):
			continue
		if option.startswith("CMAKE_"):
			continue
		if "_EXECUTABLE" in option:
			continue
		if option.startswith('pkgcfg_'):
			continue
		info = lines[i].replace("// ","")
		ret_map[option] = info
	return ret_map


basic = lines2map(basic)
advanced = lines2map(advanced)

for k in basic:
	if k in advanced:
		del advanced[k]

modified = True
while modified:
	modified = False
	for k in advanced:
		if k.startswith('CMAKE_'):
			del advanced[k]
			modified = True
			break

def pretty_print_map(title, map):
	print("####",title)
	print("| %-30s | %-10s | %-80s | %-30s |" % ("Option","Type","Description","Defaut Value"))
	print("|%s|%s|%s|%s|" % ('-'*32,'-'*12,'-'*82,'-'*32))
	for k in map:
		args = k.split(':')
		args.append(map[k])
		args += args[1].split('=')
		print("| %30s | %10s | %-80s | %-30s |" %(args[0],args[3],args[2],args[4]))
	print()

pretty_print_map("Basic Options",basic)
pretty_print_map("Advanced Options",advanced)

