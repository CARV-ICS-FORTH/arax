#!/bin/python

import os
import sys
import copy
import glob
import matplotlib.pyplot as plt
import subprocess

#CONFIG
SAVE = True
CLEANUP = True
NORMALIZE = False
HATCH = False
PALETTE = False

funcs = glob.glob("*.hdr")
funcs = [ s[0:-4] for s in funcs]

graphs = []

hdr = ".hdr"
brk = ".brk"

graphs = []

def parseLine(line):
	line = line.strip().strip(',')
	line = line.split(',')
	for i in range(0,len(line)):
		line[i] = line[i].strip()
	return line

type2str = {}
exec(open("accel_types.py").read())

def fo(fname):
	for hl,bl in zip(open(fname+hdr).readlines(),open(fname+brk).readlines()):
		head = parseLine(hl)
		braek = parseLine(bl)
		braek[1] = braek[1].replace('CPU','')
		braek[1] = braek[1].replace('GPU','')
		braek[0] = type2str[braek[0]]
		graphs.append([head,braek,len(head)])

def genColor(index,count):
	if PALETTE:
		palette = eval(open("palette.py").read())
		return palette[index%len(palette)]
	cols = []
	for i in range(0,count):
		cols.append([i/count,0.8,1-(i/count)])
	return cols[index]

def genHatch(index):
	if HATCH:
		hatches = ["+","O","|","X"]
		return hatches[index%len(hatches)]
	else:
		return ""

def number(val):
	power = 'u'
	if val > 1000:
		power = 'm'
		val = val / 1000
	if val > 1000:
		power = ''
		val = val / 1000
	return str(round(val,2))+power

def generateParsed(stats):
	stats[1] = stats[1][:stats[2]]
	stats[1][2] = int(stats[1][2])
	stats.append(0)
	for i in range(3,stats[2]):
		stats[1][i] = int(stats[1][i])/stats[1][2]
		stats[3] = stats[3] + stats[1][i]
	if len(stats[0]) != len(stats[1]):
		print("Header - Breakdown inconsistency at",title)
		sys.exit()
	return stats

def genPlot(stats,ptype,save=False,maximum=None):
	title = "::".join(stats[1][0:2])+"("+str(stats[1][2])+" runs Avg,total "+number(stats[3]/1000)+"s)"
	fname = "_".join(stats[1][0:2])+"_"+ptype+".eps"
	stats[0] = stats[0][3:]
	stats[1] = stats[1][3:]
	print(title,"->",fname)
	bot = 0
	bars = []
	i = 0

	plt.figure(figsize=(5,20))

	for val in stats[1]:
		val = val/1000000
		bars.append(plt.bar(0, val, 1, color=genColor(i,len(stats[1])),hatch=genHatch(i),bottom=bot))
		bot = val + bot
		i = i + 1

	for i in range(0,len(stats[0])):
		stats[0][i] = stats[0][i] + "(" + number(stats[1][i]/1000) + "s,"+number((stats[1][i]*100)/stats[3])[:-1]+"%)"
	plt.ylabel('Time(ms)')
	plt.title(title)
	if maximum is None:
		plt.ylim([0,bot])
	else:
		print("YLIM",[0,maximum])
		plt.ylim([0,maximum])
	plt.xticks([])
	lgd = plt.legend(bars, stats[0],loc='center',bbox_to_anchor=[0.5, -0.15])
	if save:
		plt.savefig(fname,bbox_extra_artists=(lgd,), bbox_inches='tight')
		plt.close()
		return fname
	else:
		plt.show()
		plt.close()

def generateGroupMap(groups):
	gmap = {}
	for group,cols in groups.items():
		for col in cols:
			gmap[col] = group
	return gmap

groups = {}
exec(open("groups.py").read())
gmap = generateGroupMap(groups);

def transformToGroup(stats):
	groups = {}

	for i in range(3,len(stats[0])):
		if stats[0][i] not in gmap:
			print("Unassigned column "+stats[0][i])
			sys.exit()
		else:
			key = gmap[stats[0][i]]
			if key in groups:
				groups[key] = groups[key] + stats[1][i]
			else:
				groups[key] = stats[1][i]
	stats[0] = stats[0][0:3]
	stats[1] = stats[1][0:3]

	total = 0
	for k,v in groups.items():
		stats[0].append(k)
		stats[1].append(v)
		total = total + v
	stats[2] = len(stats[1])
	stats[3] = total
	return stats

for func in funcs:
	fo(func)

raw_figs=[]
grp_figs=[]
stats = []
maximum = 0
for g in graphs:
	stat = generateParsed(g)
	stats.append(stat)
	print(maximum,stat[3])
	if maximum < stat[3]/(1000000):
		maximum = stat[3]/(1000000)

if NORMALIZE:
	maximum = None

print("Max value: ",maximum)

for stat in stats:
	raw_figs.append(genPlot(copy.deepcopy(stat),"Raw",SAVE,maximum))
	grouped = transformToGroup(copy.deepcopy(stat))
	grp_figs.append(genPlot(grouped,"Grouped",SAVE,maximum))

files = {}
files["raw.tex"] = raw_figs
files["grouped.tex"] = grp_figs

for file,figs in files.items():
	print("Generating: ",file)
	latex = "\\documentclass[a4paper,landscape]{article}\n\\usepackage{graphicx}\n\\usepackage{epstopdf}\n\\usepackage[top=1cm, bottom=1cm, left=1cm, right=1cm]{geometry}\n\\begin{document}\n"
	for fig in figs:
		latex = latex + "\t\\includegraphics[height=0.99\\textheight,width="+str(0.9/len(figs))+"\\textwidth]{"+fig+"}\n"
	latex = latex + "\end{document}"
	open(file,'w').write(latex)
	subprocess.call(["pdflatex",file],stdout=open('/dev/null','w'))
subprocess.call("rm *.aux",shell=True)
subprocess.call("rm *.log",shell=True)
subprocess.call("rm *.tex",shell=True)
subprocess.call("rm *to.pdf",shell=True)
subprocess.call("rm *.eps",shell=True)

