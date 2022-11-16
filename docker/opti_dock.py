#!/bin/env python3
import subprocess
import tempfile
from threading import Lock, RLock, Thread
import sys
import os
from datetime import datetime
from time import sleep
from concurrent.futures import ThreadPoolExecutor

push_executor = ThreadPoolExecutor(max_workers=1)

def pushHandler(img):
    with Timer("Push", img):
        if not system_do(LOGIN, os.environ["DOCK_PASS"]):
            return false
        img.status = "pushing"
        return system_do(PUSH + img.path.split())

current = None

BUILD = [
    "buildah",
    "bud",
    "-f",
    "-",
    "--volume",
    "/usr/lib/libcuda.so:/usr/lib/libcuda.so",
    "-t",
]
PULL = ["buildah", "pull", "-q","--policy=always"]
PUSH = ["buildah", "push"]
LOGIN = [
    "buildah",
    "login",
    "-u",
    os.environ["DOCK_USER"],
    "--password-stdin",
    os.environ["DOCK_HOST"],
]

docker_root = os.environ["DOCKER_ROOT"]
docker_tag = os.environ["DOCKER_TAG"]


def system_do(cmd, input=None):
    try:
        subprocess.check_output(
            cmd,
            input=input,
            text=True if input != None else False,
            stderr=subprocess.STDOUT,
            encoding="utf8",
        )
        return True
    except Exception as e:
        print("Command '{}' failed, log:\n".format(" ".join(cmd)))
        print("\t" + e.output.replace("\n", "\n\t"))
    return False


def normalizeDelta(delta):
    unit = ["us", "ms", "s ", "m ", "h "]
    factor = [1000000, 1000, 1, 1 / 60.0, 1 / 3600]

    for i in range(len(unit)):
        norm = delta * factor[i]
        if norm < 1000:
            return "{:6.2f} {}".format(norm, unit[i])


class Timer:
    zero_time = datetime.timestamp(datetime.now())
    cumulative_time = 0.0
    ctime_lock = Lock()

    def __init__(self, op, msg):
        self.op = op
        if isinstance(msg,str):
            self.msg = msg
        else:
            self.msg = msg.path
            msg.timer = self

    def __enter__(self):
        self.start = datetime.timestamp(datetime.now())

    def __str__(self):
        return "{:5} {} {} {} {}".format(
            self.op,
            normalizeDelta(self.stop - self.start),
            normalizeDelta(Timer.cumulative_time),
            normalizeDelta(self.stop - Timer.zero_time),
            self.msg,
        )

    def toMermaid(self):
        dec = ""
        dec = "active, " if self.op == "Push" else dec
        dec = "crit, " if self.op == "Pull" else dec
        return "{:30}: {}{}, {}".format(self.op,dec,int(self.start),int(self.stop))

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop = datetime.timestamp(datetime.now())
        with Timer.ctime_lock:
            if self.msg != "Total":
                Timer.cumulative_time += self.stop - self.start
            print(str(self),file=sys.stderr)

def sortByImportance(obj):
    childs = 1

    if type(obj) == str:
        obj = graph.getImage(obj)

    for c in obj.childs:
        childs += sortByImportance(c) % 1000

    priority = {}
    priority["pull"] = 1000000
    priority["push"] = 1000
    priority["build"] = 0

    return childs + priority[obj.type]

def thFn(obj):
    obj.doIt_()

class Image:
    def __init__(self, graph, path):
        self.path = path
        self.graph = graph
        self.parents = set()
        self.childs = []
        self.type = "pull"
        self.commands = []
        self.status = "pending"
        self.timer = None

    def push(self):
        push_executor.submit(pushHandler,self)
        return True

    def pull(self):
        with Timer("Pull", self):
            self.status = "fetching"
            if self.path != "scratch":
                return system_do(PULL + [self.path])
        return True

    def build(self):
        self.status = "building"
        dock_cmds = "\n".join(self.commands)
        with Timer("Build", self):
            return system_do(BUILD + [self.path], dock_cmds)

    def doIt_(self):
        cmd = "self.{}".format(self.type)
        ret = eval(cmd)()
        self.graph.updateRoots()
        self.graph.good = ret

    def doIt(self):
        Thread(target=thFn, args=[self]).run()

    def addCommand(self, cmd):
        self.commands.append(cmd)

    def addChild(self, child):
        if not child in self.childs:
            self.childs.append(child)
            child.addParent(self)

    def addParent(self, parent):
        if not parent in self.parents:
            self.parents.add(parent)
            parent.addChild(self)
        self.graph.removeRoot(self.path)
        self.type = "build"

    def uid(self):
        return hex(abs(hash(self.path)))[1:]

    def isRoot(self):
        return len(self.parents) == 0

    def remove(self):
        del self.graph.images[self.path]
        assert self.isRoot()
        for child in self.childs:
            child.parents.remove(self)
        self.graph.completed.append(self)

    def dotDefinition(self):
        misc = ""
        if len(self.parents) == 0:
            misc += ' shape="record"'
        return '{:20} [label="{}:{}({})"{}];'.format(
            self.uid(), self.type, self.path, sortByImportance(self), misc
        )


class ImageGraph:
    def __init__(self):
        self.images = {}
        self.completed = []
        self.roots = set()
        self.lock = RLock()
        self.good = True

    def getImage(self, path):
        with self.lock:
            if not path in self.images:
                self.images[path] = Image(self, path)
                self.roots.add(path)
            return self.images[path]

    def removeRoot(self, root):
        with self.lock:
            if root in self.roots:
                self.roots.remove(root)

    def toDOT(self):
        print("digraph test {")
        for img in self.images:
            print(" ", self.getImage(img).dotDefinition())
        print()
        for img in self.images:
            for chld in self.images[img].childs:
                print("  {} -> {}; ".format(graph.getImage(img).uid(), chld.uid()))
        print("}")

    def toMermaid(self):
        print("gantt")
        print("dateFormat X")
        print("section Build")
        sections = {}

        for img in self.completed:
            section = img.path.split()[0].replace(':','$')
            if not section in sections:
                sections[section] = ""
            sections[section] += img.timer.toMermaid()+'\n'

        for section in sections:
            print("section " + section.replace("docker.io",""))
            print(sections[section])

    def hasRoots(self):
        with self.lock:
            return len(self.roots) > 0

    def removeImage(self, image):
        with self.lock:
            self.removeRoot(image)
            self.getImage(image).remove()

    def updateRoots(self):
        with self.lock:
            for img in self.images:
                if self.getImage(img).isRoot():
                    self.roots.add(img)

    def popNext(self):
        with self.lock:
            if len(self.roots) > 0:
                next = max(self.roots, key=sortByImportance)
                img = self.getImage(next)
                self.removeImage(img.path)
                return img
            else:
                return None

    def Execute(self):
        timeline = []

        while len(self.images) > 0 and self.good:
            victim = self.popNext()
            if victim != None:
                victim.doIt()
            else:
                sleep(0.1)

        print("\n".join(timeline))

        return self.good


def NOP(parts, graph):
    pass


def FROM(parts, graph):
    parent = graph.getImage(parts[1])
    child = graph.getImage(parts[3])
    child.addParent(parent)
    global current
    current = child
    pass


def COPY(parts, graph):
    for part in parts:
        if part.startswith("--from"):
            dep = part[7:]
            current.addParent(graph.getImage(dep))
    pass


ENV = NOP
RUN = NOP
WORKDIR = NOP
ENTRYPOINT = NOP

reps = {}
reps["DOCK_HOST/accelerators"] = "docker.io/nvidia"

with open(sys.argv[1]) as df:
    graph = ImageGraph()
    for line in df:
        for rep in reps:
            if rep in line:
                line = line.replace(rep, reps[rep])
        parts = line.split()
        if line.startswith("#@"):
            push_path = (
                line[2:].strip() if line[2:].strip() else docker_root + current.path
            )
            push_img = graph.getImage(
                "{} {}:{}".format(current.path, push_path, docker_tag)
            )
            current.addChild(push_img)
            push_img.type = "push"
        if not line.startswith("#") and len(parts) > 1:
            eval(parts[0])(parts, graph)
            current.addCommand(" ".join(parts))

graph.toDOT()

print(
    "{:5} {:9} {:9} {:9} {}".format("Op", "I.Time", "C.Time", "T.Time", "Image"),
    file=sys.stderr,
)

ran = False
with Timer("Total", "Total"):
    ran = graph.Execute()
    push_executor.shutdown()

graph.toMermaid()

sys.exit(0 if ran else 1)
