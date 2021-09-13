#!/usr/bin/python3

import subprocess
from datetime import date

accs = {}

aliases = {
	'Foivos S. Zakkak <foivos@zakkak.net>' : 'Foivos S. Zakkak <zakkak@ics.forth.gr>',
	'alexis <akavroulakis@gmail.com>' : 'Kavroulakis Alexandros <kavros@ics.forth.gr>',
	'dmcbeing <dmcbeing@hotmail.com>' : 'Stelios Mavridis <mavridis@ics.forth.gr>',
	'root <root@kurosawa.ics.forth.gr>' : 'Stelios Mavridis <mavridis@ics.forth.gr>',
	'mavridis <mavridis@ics.forth.gr>' : 'Stelios Mavridis <mavridis@ics.forth.gr>',
	'SAVVAS KIAOURTZIS <savaskia@localhost.localdomain>' : 'Savas Kiaourtzis <savaskia@ics.forth.gr>',
	'savaskia <savaskia@ics.forth.gr>' : 'Savas Kiaourtzis <savaskia@ics.forth.gr>',
	'manospavl <manospavl@ratlab.ics.forth.gr>' : 'Manos Pavlidakis <manospavl@ics.forth.gr>',
	'manos pavlidakis <manospavl@ics.forth.gr>' : 'Manos Pavlidakis <manospavl@ics.forth.gr>',
	'samatas <samatas@ics.forth.gr>': 'Dimitris Samatas <samatas@ics.forth.gr>'
}

output=subprocess.check_output("git log --format=short", shell=True, encoding='UTF8')

for line in output.split('\n'):
	if line.startswith('Author: '):
		author = line.split(':')[1].strip()

		acc = author

		if acc in aliases:
			acc = aliases[acc]

		if acc not in accs:
			accs[acc] = 1
		else:
			accs[acc] += 1

hist = []
			
for acc in  accs:
	hist.append((acc, accs[acc]))

hist = sorted(hist)

print("# Contributors[^contribs]")
print("[^contribs]: As of %s" % (date.today()))
print()
print("| %25s | %25s |" % ('Name',"Email at ics.forth.gr"))
print("|-%s-|-%s-|" % ('-'*25,'-'*25))
for commiter in hist:
	vals = commiter[0].replace('>','').split('<')
	print("| %25s | %25s |" % (vals[0],vals[1].split('@')[0]))
