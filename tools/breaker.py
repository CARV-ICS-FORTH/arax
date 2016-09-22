#!/bin/python

import os
import sys
import glob
import matplotlib.pyplot as plt


#CONFIG
SAVE = False


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
type2str['0'] = 'ANY'
type2str['1'] = 'Gpu'
type2str['2'] = '???'
type2str['3'] = 'Cpu'

def fo(fname):
	for hl,bl in zip(open(fname+hdr).readlines(),open(fname+brk).readlines()):
		head = parseLine(hl)
		braek = parseLine(bl)
		braek[1] = braek[1].replace('CPU','')
		braek[1] = braek[1].replace('GPU','')
		braek[0] = type2str[braek[0]]
		graphs.append([head,braek,len(head)])

def genColor(index,count):
	cols = []
	for i in range(0,count):
		cols.append([i/count,0.8,1-(i/count)])
	return cols[index]

def genHatch(index):
	hatches = ["+","O","|","X"]
	return hatches[index%len(hatches)]

def number(val):
	power = 'u'
	if val > 1000:
		power = 'm'
		val = val / 1000
	return str(round(val,2))+power

def genRawPlot(stats,save=False):
	fname = "_".join(stats[1][0:2])+"_raw.eps"
	stats[1] = stats[1][:stats[2]]
	stats[1][2] = int(stats[1][2])
	stats.append(0)
	for i in range(3,stats[2]):
		stats[1][i] = int(stats[1][i])/stats[1][2]
		stats[3] = stats[3] + stats[1][i]
	title = "::".join(stats[1][0:2])+"("+str(stats[1][2])+" runs Avg,total "+number(stats[3]/1000)+"s)"
	if len(stats[0]) != len(stats[1]):
		print("Header - Breakdown inconsistency at",title)
		sys.exit()

	stats[0] = stats[0][3:]
	stats[1] = stats[1][3:]
	print(title,"->",fname)
	bot = 0
	bars = []
	i = 0

	for val in stats[1]:
		val = val/1000000
		print(val)
		bars.append(plt.bar(0, val, 1, color=genColor(i,len(stats[1])),hatch=genHatch(i),bottom=bot))
		bot = val + bot
		i = i + 1

	for i in range(0,len(stats[0])):
		stats[0][i] = stats[0][i] + "(" + number(stats[1][i]/1000) + "s,"+number((stats[1][i]*100)/stats[3])[:-1]+"%)"
	plt.ylabel('Time(ms)')
	plt.title(title)
	plt.yticks([0,bot])
	plt.legend(bars, stats[0])
	if save:
		plt.savefig(fname)
	else:
		plt.show()


for func in funcs:
	fo(func)

for g in graphs:
	genRawPlot(g,SAVE)

