import gitlab
import sys
import os

host = 'https://carvgit.ics.forth.gr'
token = os.environ['CID_TOKEN']
project = os.environ['CI_PROJECT_PATH']
branch = os.environ['CI_BUILD_REF_NAME']
commit_sha = os.environ['CI_COMMIT_SHA']
proj_url = os.environ['CI_PROJECT_URL']
user_email = os.environ['GITLAB_USER_EMAIL']

def writeMessage(commit,msg):
	commit.comments.create({'note': msg})

def t(msg='-'*20):
	return "| %-20s " % (msg,)

def StatusMark(status):
	if status == 'success':
		return t('{+success+}')
	elif status == 'failed':
		return t('{-success-}')
	else:
		return t(status)

with gitlab.Gitlab(host, private_token=token) as gl:
	vt = gl.projects.get(project)
	commits = vt.commits.list(ref_name=branch)
	tables = []
	coverages = []
	failed = False
	for commit in commits[:2]:
		statuses = commit.statuses.list()
		coverage = None
		table = t("Stage")+t("Status")+t("Link")+"|   \n"
		table += t()+t()+t()+"|   \n"
		for status in statuses:
			print(status,"\n")
			if coverage is None:
				coverage = status.coverage
			if status.started_at != None and status.status != 'running':
				table += t(status.name)+StatusMark(status.status)+t("<a href='"+proj_url+"/-/jobs/"+str(status.id)+"'>View</a>")+"|   \n"
			if status.status == 'failed':
				failed = True
		tables.append(table)
		if coverage == None:
			coverage = 0
		coverages.append(coverage)
		
	msg = ""
	
	user = "@" + user_email.split('@')[0]
	
	if failed:
		msg += "# %s your commit did not pass the tests!  \n" % (user,)
	else:
		msg += "# %s your commit passed the tests!  \n" % (user,)

	cov_delta = coverages[0]-coverages[1]
	msg += "#### Coverage: %6.2f%% (%6.2f%%)  \n" % (coverages[0],cov_delta)
	msg += tables[0]
	print(msg)
	writeMessage(commits[0],msg)
	
