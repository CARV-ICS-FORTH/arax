#!.miniconda/bin/python 
import inquirer
import sys

def genScope(scope):
  if len(scope) == 1 and "Other" in scope:
    return ""
  if len(scope) == 0:
    return ""
  else:
    ret = "("
    sep = ""
    for s in scope:
      ret += sep + s
      sep = ','
    ret += ")"
    return ret

def notEmpty(anwsers, current):
  if current.strip() == "":
    return False
  return True

def warnMulti(anwsers, current):
  if len(current) == 0:
    raise inquirer.errors.ValidationError('', reason='Select at least one')
    return False
  if len(current) > 1:
    if not inquirer.confirm("Commits should affect a single scope, procced anyway?", default=False):
      print("Commit aborted, split your commit to individual scopes")
      sys.exit(1)
  return True

questions = [
  inquirer.List('Type', message="Commit Type", choices=[ 'feat', 'fix', 'test', 'build', 'ci', 'docs', 'perf', 'refactor','style']),
  inquirer.Checkbox('Scope','What changed(select with space)',choices=['arch', 'async','core', 'utils', 'JVTalk', 'Other'],validate=warnMulti),
  inquirer.Text('Title', message="Single line description", validate=notEmpty),
  inquirer.Editor('Description', message="Larger description", validate=notEmpty)
]

answers = inquirer.prompt(questions)
old_msg = open(sys.argv[1],'r').read()
with open(sys.argv[1],'w') as msg:
  msg.write("%s%s: %s\n\n%s\n" % (answers['Type'],genScope(answers['Scope']),answers['Title'],answers['Description'].strip()))
  msg.write("\n### Last chance to review/abort your commit! ###\n")
  msg.write(old_msg)
