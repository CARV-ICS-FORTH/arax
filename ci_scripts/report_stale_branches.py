import gitlab
import os

host = 'https://carvgit.ics.forth.gr'
token = os.environ['CID_TOKEN']
project = os.environ['CI_PROJECT_PATH']
current_branch = os.environ['CI_BUILD_REF_NAME']

rows = {}

def rebaseIssue(proj, branch, open):
    title = "Branch " + branch + " needs rebase"
    owner = proj.commits.list(ref_name=branch,page=1,per_page=1)[0].committer_email.split('@')[0]
    desc = "@" + owner + " please sync/rebase your branch " + branch + " with master, or delete if unused."
    for issue in proj.issues.list(all=True):
        if issue.title == title:
            issue.description = desc
            issue.state_event = 'reopen' if open else 'close'
            issue.save()
            break
    else:
        if open:
            proj.issues.create({'title':title,'description':desc})

with gitlab.Gitlab(host, private_token=token) as gl:
    vt = gl.projects.get(project)
    master_top_two = vt.commits.list(ref_name="master",per_page=2,page=1)
    master_head = master_top_two[0].id
    stale_commits = set()
    stale_commits.add(master_top_two[1].id)
    for branch in vt.branches.list(as_list=False):
        commits = vt.commits.list(ref_name=branch.name,as_list=False)
        cpath = set()
        stale = False
        for commit in commits:
            if commit.id == master_head:
                break
            if commit.id in stale_commits:
                stale_commits.update(cpath)
                stale = True
                break
            else:
                cpath.add(commit.id)
        else:
            stale_commits.update(cpath)
            stale = True
        owner = vt.commits.list(ref_name=branch.name,page=1,per_page=1)[0].committer_email.split('@')[0]
        rows[branch.name] = [owner,stale]

    for row in rows:
        print(row.ljust(30),rows[row][0].ljust(20),"needs rebase!" if rows[row][1] else "is in sync")
        rebaseIssue(vt, row, rows[row][1])
