#!.miniconda/bin/python
import inquirer
import sys

def genScope(scope):
  if scope == None:
    return ""
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

def genDesc(desc):
  desc = desc.strip()
  if desc != "":
    desc = "\n" + desc + "\n"
  return desc

def checkTitle(anwsers, current):
  if current.strip() == "":
    raise inquirer.errors.ValidationError('', reason='Title can\'t be empty')
    return False
  if len(current) > 72:
    raise inquirer.errors.ValidationError('', reason='Title must be shorter than 74 characters')
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

"""Dont ask for scope if type is not: feat,fix,test or perf"""
def hideScope(anwsers):
  show = ['feat', 'fix','test' , 'perf']
  if anwsers['Type'] in show:
    return False
  return True

questions = [
  inquirer.List('Type', message="Commit Type", choices=[ 'feat', 'fix', 'test', 'chore', 'docs', 'perf', 'refactor','style']),
  inquirer.Checkbox('Scope','What changed(select with space)',choices=['arch', 'async','core', 'utils', 'JVTalk', 'Other'],validate=warnMulti,ignore=hideScope),
  inquirer.Text('Title', message="Single line description", validate=checkTitle),
  inquirer.Editor('Description', message="Larger description")
]

answers = inquirer.prompt(questions)

old_msg = open(sys.argv[1],'r').read()
with open(sys.argv[1],'w') as msg:
  msg.write("%s%s: %s\n%s" % (answers['Type'],genScope(answers['Scope']),answers['Title'].strip(),genDesc(answers['Description'])))
