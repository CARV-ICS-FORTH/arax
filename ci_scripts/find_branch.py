import gitlab
import sys
import os

host = 'https://carvgit.ics.forth.gr'
token = os.environ['CID_TOKEN']

project = os.environ['CI_PROJECT_PATH']
branch = os.environ['CI_BUILD_REF_NAME']
target_repo = sys.argv[1]	# Repo to clone from with closest branch

def hasBranch(repo, branch):
	try:
		proj = gl.projects.get(repo)
		proj.branches.get(branch)
		return True
	except:
		return False

def parentBranch(repo, branch):
	vt = gl.projects.get(project)
	vt_branches = vt.branches.list()
	cntrl = gl.projects.get(repo)

	candidates = []

	for vt_branch in vt_branches:
		try:
			cntrl.branches.get(vt_branch.name)
			candidates.append(vt_branch.name)
		except:
			pass
		
	if len(candidates) < 3:
		if len(candidates) == 2:
			for cand in candidates:
				if cand != 'master':
					return cand
		return candidates[0]
	else:
		return None

with gitlab.Gitlab(host, private_token=token) as gl:
	while branch != None and not hasBranch(target_repo, branch):
		branch = parentBranch(target_repo, branch)
	if branch != None:
		print(branch)
	else:
		print("NO_PARENT_BRANCH_FOUND")
