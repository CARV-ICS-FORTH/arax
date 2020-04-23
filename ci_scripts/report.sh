import gitlab
import sys

gl = gitlab.Gitlab('https://carvgit.ics.forth.gr', job_token=os.environ['CI_JOB_TOKEN'])

projects = gl.projects.list()
for project in projects:
    print(project)
