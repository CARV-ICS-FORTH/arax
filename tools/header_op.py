# Header dependecy graph generator
#!/bin/python
import glob, os
import hashlib

os.chdir("../src/")
files=[]
for filename in glob.iglob('**/*.c', recursive=True):
    files.append(filename)
for filename in glob.iglob('**/*.h', recursive=True):
    files.append(filename)

def hashit(val):
	return "MD"+hashlib.md5(val.encode()).hexdigest();

print("digraph header_deps {");
nodes=[]
targets=[]
aliases=[]
for file in files:
	parts=file.split('/')
	alias=""
	for part in parts[::-1]:
		alias = part+"/"+alias
		alias = alias.strip('/')
		if alias != file:
			aliases.append((alias,file))
			targets.append(file)
		nodes.append(alias)
for file in files:
	for line in open(file):
		if "#include" in line:
			line = line[8:].strip()
			line = line.replace("<","")
			line = line.replace(">","")
			line = line.replace("\"","")
			if line in nodes:
				print(hashit(file),"->",hashit(line)+"; // ",file,"->",line);
				targets.append(file)
				targets.append(line)

for target in targets:
	print(hashit(target),"[label=\""+target+"\"]")
for alias,file in aliases:
	if alias in targets:
		print(hashit(alias),"->",hashit(file),"[label=\"sortof\"]")

print("}");

#./header_op.py | dot -Teps > heads.eps
