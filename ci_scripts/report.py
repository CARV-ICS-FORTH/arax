import datetime
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
pipeline_id = os.environ['CI_PIPELINE_ID']

def writeMessage(commit,msg):
	commit.comments.create({'note': msg})

def t(msg='-'*20):
	return "| %-20s " % (msg,)

def c():
	return "| :%-18s: " % ('-'*18,)

def StatusMark(status):
	if status == 'success':
		return t('{+success+}')
	elif status == 'failed':
		return t('{-failed-}')
	else:
		return t(status)

def parseTime(t):
	if type(t) == type(""):
		return datetime.datetime.strptime(t[:21],'%Y-%m-%dT%H:%M:%S.%f')
	else:
		return t

def statDuration(status):
	return calcDuration(status.started_at,status.finished_at)

def pipeDuration(pipe):
	return calcDuration(pipe.created_at,pipe.updated_at)

def calcDuration(start,end):
	start = parseTime(start)
	end = parseTime(end)

	delta = end-start
	ret = str(delta).rstrip('0')
	prev = None

	while ret != prev:
		prev = ret
		ret = ret.lstrip('0:')

	return ret

with gitlab.Gitlab(host, private_token=token) as gl:
	vt = gl.projects.get(project)
	commits = vt.commits.list(ref_name=branch)
	tables = []
	coverages = []
	fails = []

	pipeline = vt.pipelines.get(pipeline_id)

	print(pipeline)

	for commit in commits[:2]:
		failed = False
		statuses = commit.statuses.list()
		coverage = None
		table = t("Stage")+t("Status")+t("Duration")+t("Link")+"|   \n"
		table += t()+t()+c()+t()+"|   \n"
		last_job = parseTime(pipeline.created_at)
		for status in statuses:
			if coverage is None:
				coverage = status.coverage
			if status.status == 'failed':
				failed = True
			if status.started_at != None and status.status != 'running':
				last_job = max(last_job,parseTime(status.finished_at))
				table += t(status.name)+StatusMark(status.status)+t(statDuration(status))+t("<a href='"+proj_url+"/-/jobs/"+str(status.id)+"'>View</a>")+"|   \n"

		table += t("Overall")+StatusMark("failed" if failed else "success")+t(calcDuration(pipeline.created_at,last_job))+t("<a href='"+proj_url+"/pipelines/"+str(pipeline.id)+"'>View</a>")+"|   \n"
		tables.append(table)
		if coverage == None:
			coverage = 0
		coverages.append(coverage)
		fails.append(failed)

	msg = ""

	user = "@" + user_email.split('@')[0]

	if fails[0] == True:
		msg += "# Commit failed the tests!  \n"
	else:
		msg += "# Commit passed the tests!  \n"

	msg += "#### Commit: %s  \n" % (commit_sha,)
	msg += "#### User: %s  \n" % (user,)
	msg += "#### Branch: %s  \n" % (branch,)
	cov_delta = coverages[0]-coverages[1]
	msg += "#### Coverage: %6.2f%% (%6.2f%%)  \n" % (coverages[0],cov_delta)
	msg += tables[0]
	print(msg)
	writeMessage(commits[0],msg)

