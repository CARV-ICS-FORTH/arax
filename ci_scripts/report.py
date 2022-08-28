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
vc_branch = os.environ['ARAX_CNTRL_BRANCH']
BOT_MSG = "This is a bot generated comment"

def writeMessage(commit,msg):
	for disc in commit.discussions.list():
		for note in disc.attributes['notes']:
			if BOT_MSG in note['body']:
				disc.notes.delete(note['id'])
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

def view(url):
	return "<a href='%s'>View</a>" % (url)

def durs(n):
	return "%.2fs" % (n)

def pipeStatus(good,fail):
	return t("&#x2714;:%d &#x274C;:%d" % (good,fail))

def jobStatus(job):
	dec = "&#x2714;" if job.status == "success" else "&#x274C;"
	return t("%s{+%s+}%s" % (dec,job.status,dec))

with gitlab.Gitlab(host, private_token=token) as gl:
	vt = gl.projects.get(project)
	commits = vt.commits.list(ref_name=branch)
	coverages = []
	display_status = ['success','failed']

	pipeline = vt.pipelines.get(pipeline_id)

	table = t("Stage")+t("Status")+t("Duration")+t("Link")+"|   \n"
	table += t()+t()+c()+t()+"|   \n"

	total_dur = 0
	good_jobs = 0
	bad_jobs = 0

	job_map = {}

	for job in pipeline.jobs.list():
		if job.status in display_status:
			table += t(job.name)+jobStatus(job)+t(durs(job.duration))+t(view(job.web_url))+"|   \n"
			total_dur += job.duration

			if not job.name in job_map:
				job_map[job.name] = False

			if job.status == 'failed':
				bad_jobs += 1
			else:
				good_jobs += 1
				job_map[job.name] = True

	all_good = True

	for job in job_map:
		if not job_map[job]:
			all_good = False

	table += t("Total")+pipeStatus(good_jobs,bad_jobs)+t(durs(total_dur))+t(view(pipeline.web_url))+"|   \n"

	msg = ""

	user = "@" + user_email.split('@')[0]

	status = "passed" if all_good else "failed"

	msg += "# Commit " + status + " the tests! [^1]  \n"

	msg += "#### Commit: %s  \n" % (commit_sha,)
	msg += "#### User: %s  \n" % (user,)
	msg += "#### Arax Branch: %s  \n" % (branch,)
	msg += "#### Controller Branch: %s  \n" % (vc_branch,)
	msg += "#### Coverage: %6.2f%%  \n" % (float(pipeline.coverage) if pipeline.coverage != None else 0.0)
	msg += table
	print(msg)
	msg += "[^1]: " + BOT_MSG
	writeMessage(commits[0],msg)

