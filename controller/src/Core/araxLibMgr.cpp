#include "AraxLibMgr.h"
#include "Utilities.h"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sys/inotify.h>
#include <sys/types.h>
#include <unistd.h>

std::string commonPrefix(const std::string &a, const std::string &b) {
  auto len = std::min(a.find_last_of('_'), b.find_last_of('_'));
  if (len == std::string::npos)
    return "";
  len++;
  if (a.substr(0, len) == b.substr(0, len))
    return a.substr(0, len);
  return "";
}

std::string differentPostfix(size_t c, const std::string &a,
                             const std::string &b) {
  return "<" + a.substr(c) + "," + b.substr(c) + ">";
}

std::ostream &operator<<(std::ostream &os, AraxLibMgr ::KernelMap km) {
  std::map<std::string, size_t> str_len;
  size_t col_width = 1;
  for (auto arch : km.krnl_archs) {
    os.width(5);
    os << arch;
    os << ESC_CHR(ANSI_BOLD);
    os << ESC_CHR(ANSI_RST) << " |";
    str_len[arch] = std::max(arch.length(), (size_t)5);
    col_width += str_len[arch] + 2;
  }
  AraxLibMgr ::KernelMap temp;
  temp.krnl_archs = km.krnl_archs;

  bool merged;
  // Find common post fixes and join - only joins pairs for now
  for (auto itr = km.krnl_state.begin(); itr != km.krnl_state.end(); itr++) {
    auto next = itr;
    next++;
    if (next == km.krnl_state.end()) {
      temp.krnl_state[itr->first] = itr->second;
      break;
    }
    merged = false;
    if (itr->second == next->second) {
      std::string cp = commonPrefix(itr->first, next->first);
      std::string dp = differentPostfix(cp.size(), itr->first, next->first);
      if ((itr->first + next->first).size() + 2 > (cp + dp).size()) {
        merged = true;
        temp.krnl_state[cp + dp] = itr->second;
        itr++;
      }
    }
    if (!merged)
      temp.krnl_state[itr->first] = itr->second;
  }
  km = temp;
  // Compact kernels a bit
  auto itr = km.krnl_state.begin();
  while (itr != km.krnl_state.end()) {
    auto cur = itr;
    auto next = itr;
    next++;
    if (next == km.krnl_state.end()) {
      itr++;
      continue;
    }
    if (cur->second == next->second) {
      if ((cur->first + ", " + next->first).size() + col_width < 75) {
        km.krnl_state[cur->first + ", " + next->first] = next->second;
        km.krnl_state.erase(cur->first);
        km.krnl_state.erase(next->first);
        itr = km.krnl_state.begin();
        continue;
      }
    }
    itr++;
  }
  os << " Operation" << std::endl;
  for (auto krnl : km.krnl_state) {
    for (auto arch : km.krnl_archs) {
      if (krnl.second.count(arch)) {
        if (krnl.second[arch]) {
          os << ESC_CHR(ANSI_GREEN);
          os.width(str_len[arch]);
          os << "GOOD" << ESC_CHR(ANSI_RST) << " |";
        } else {
          os << ESC_CHR(ANSI_RED);
          os.width(str_len[arch]);
          os << "FAIL" << ESC_CHR(ANSI_RST) << " |";
        }
      } else {
        os << ESC_CHR(ANSI_BLUE);
        os.width(str_len[arch]);
        os << "----" << ESC_CHR(ANSI_RST) << " |";
      }
    }
    os << " " << krnl.first << std::endl;
  }

  return os;
}

void AraxLibMgr ::KernelMap ::addKernel(std::string arch, std::string krnl,
                                        bool state) {
  if (krnl_state.count(krnl) && krnl_state[krnl].count(arch)) {
    std::cerr << ESC_CHR(ANSI_YELLOW) << arch << "::" << krnl
              << " duplicate registration!" << ESC_CHR(ANSI_RST) << "\n";
  }
  krnl_state[krnl][arch] = state;
  krnl_archs.insert(arch);
}

AraxLibMgr::AraxLibMgr() : run(true) { inotify_fd = inotify_init(); }

bool AraxLibMgr::loadFolder(string lib_path, KernelMap &km, bool silent) {
  string lib;
  DIR *dir;
  dirent *itr;

  if (lib_paths.count(lib_path))
    return true;

  dir = opendir(lib_path.c_str());

  if (!dir) {
    if (!silent)
      cerr << "Path \'" << lib_path << "\' could not be opened." << endl;
    return false;
  }

  lib_fd_paths[inotify_add_watch(inotify_fd, lib_path.c_str(),
                                 IN_CLOSE_WRITE)] = lib_path;

  lib_paths.insert(lib_path);

  do {
    itr = readdir(dir);
    if (itr) {
      if (itr->d_name[0] == '.')
        continue;

      lib = lib_path + "/" + itr->d_name;
      if (!loadFolder(lib, km, true))
        loadLibrary(lib, km);
    }
  } while (itr);

  closedir(dir);
  return true;
}
bool AraxLibMgr::loadLibrary(string lib_file, KernelMap &km) {
  if (lib_file.rfind(".so") == string::npos)
    return false;

  if (libs.count(lib_file))
    return true;

  void *handle = dlopen(lib_file.c_str(), RTLD_NOW); /* Fail now */
  AraxProcedureDefinition *defs;

  if (!handle) {
    cerr << __func__ << " => " << ESC_CHR(ANSI_RED) << dlerror()
         << ESC_CHR(ANSI_RST) << endl;
    return false;
  }

  /* We have a library, but is it a Arax lib? */
  defs = (AraxProcedureDefinition *)dlsym(handle, "arax_proc_defs");

  if (!defs) {
    dlclose(handle);
    return false;
  }
  libs[lib_file] = handle;

  while (defs->name) {
    if (defs->max_type > ARAX_ACCEL_TYPES)
      cerr << "Warning: " << defs->name << "() in " << lib_file
           << " compiled with future version!" << endl;
    if (defs->type >= ARAX_ACCEL_TYPES) {
      cerr << "Error: " << defs->name << "() in " << lib_file
           << " targeting for unknown accelerator,skiping!" << endl;
      defs++;
      continue;
    }
    void *pntr = (void *)(defs->functor);
    arax_proc *proc;

    proc = arax_proc_register(defs->name);
    arax_proc_set_functor((arax_proc_s *)proc, defs->type, (AraxFunctor *)pntr);
    km.addKernel(arax_accel_type_to_str(defs->type), defs->name, !!proc);
    defs++;
  }

  return true;
}

void AraxLibMgr::unloadLibraries(arax_pipe_s *pipe) {

  for (ProcSet::iterator itr = procs.begin(); itr != procs.end(); itr++) {
    cerr << "Unregistering: " << ((arax_proc_s *)*itr)->obj.name << "()"
         << endl;
    arax_object_ref_dec((arax_object_s *)*itr);
  }
  procs.clear();
  for (Str2VpMap::iterator itr = libs.begin(); itr != libs.end(); itr++)
    dlclose(itr->second); /*TODO --> this call crashes when using streams*/
  libs.clear();
}

void AraxLibMgr::startRunTimeLoader() {
  rtld = std::thread(rtld_handler, this);
}

union InotifyEvent {
  struct inotify_event event;
  char padd[NAME_MAX + 1];
};

void AraxLibMgr::rtld_handler(AraxLibMgr *libmgr) {
  set_thread_name("RtLb");
  while (libmgr->run) {
    KernelMap km;
    InotifyEvent event;

    read(libmgr->inotify_fd, &event, sizeof(event));

    libmgr->loadLibrary(
        libmgr->lib_fd_paths[event.event.wd] + "/" + event.event.name, km);
  }
}

AraxLibMgr::~AraxLibMgr() {
  run = false;
  // Create a fake library file 'run.false' to wakeup rtld thread
  { std::ofstream signal(lib_fd_paths.begin()->second + "/run.false"); }
  rtld.join();
  if (procs.size() || libs.size())
    cerr << "AraxLibMgr still has " << procs.size()
         << " procedures"
            "and "
         << libs.size() << " libraries loaded!" << endl;
}
