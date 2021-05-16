import sys
import os

def red(line):
	return '%c[31m%s%c[0m' % (27,line,27)

def orange(line):
	return '%c[346m%s%c[0m' % (27,line,27)

def green(line):
	return '%c[32m%s%c[0m' % (27,line,27)

class Struct:
	def __init__(self):
		self.name = None
		self.lines = []
		self.holes = 0
	def addLine(self,line):
		self.lines.append(line)
		if self.name == None:
			self.name = line.split()[1]
	def closed(self):
		return len(self.lines) > 1 and "}" in self.lines[-1]
	def __str__(self):
		ret = green("[" + self.name.center(78) + "]") + "\n"
		for line in self.lines:
			if "hole" in line:
				line = red(line)
			if "cacheline" in line:
				line = orange(line)
			ret += line
		return ret
			
structs = []
structs.append(Struct())

pahole = os.popen("pahole -H 1 ./build/libvine.so")


for line in pahole.readlines():
	if structs[-1].closed():
		structs.append(Struct())
	structs[-1].addLine(line)

ignore_list = [
	"_IO_FILE"
]

for s in structs:
	if s.name not in ignore_list:
		print(s)

if len(structs):
	print(red("There are %d structs with holes" % len(structs)))
#	sys.exit(-1)
