// isis - a minimal ISIS-II on the qkz80 8080 core, enough to run Intel's
// PL/M-80 V3.1, LINK and LOCATE.
//
//   isis PROGRAM [arguments...]
//
// runs the ISIS absolute object file PROGRAM (for example PLM80) with the
// command line "PROGRAM arguments...", the way the ISIS CLI would.  Every
// :Fn: drive is the current directory, and a name is looked up in it without
// regard to case; :CI: reads standard input and :CO: writes standard output,
// :BB: is the byte bucket and :LP: appends to LP.OUT.  scripts/intel_oracle.py
// runs Intel's compiler through it, the way DRI's P.SUB did under ISX.
//
// Intel's binaries are not part of this repository.  They are on DRI's MP/M
// II work disk (mpm2src/PLM_WORK), which the oracle finds by itself.
//
// Environment: ISIS_TRACE=1 traces the ISIS calls to standard error;
// MEMTOP=hhhh sets the top of memory MEMCK reports (default F6EF, a 62K
// ISIS system, which is where DRI's builds put the top).
//
// Exit status: 0 when the program calls EXIT or jumps to 0000H, 2 when the
// program cannot be loaded, 3 when it runs past the instruction limit.
//
// Copyright (C) 2026 Aaron Wohl.  GNU General Public License v3 or later,
// as the rest of uplm80.

#include "qkz80.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <strings.h>
#include <unistd.h>
#include <vector>

static unsigned char *M;
static qkz80_cpu_mem memory;
static qkz80 cpu(&memory);
static bool trace_isis = false;

// ISIS file table entries.  0 and 1 are the console, open from the start.
struct AFT {
  bool used = false;
  bool console_in = false, console_out = false, bucket = false;
  FILE *f = nullptr;
  std::string path;
  int access = 0;
  long pos = 0;
};
static AFT afts[32];

static std::string cmdline;  // the pending console line, with CR LF
static size_t cipos = 0;     // how much of it has been read

static unsigned w(unsigned a) { return M[a & 0xffff] | (M[(a + 1) & 0xffff] << 8); }
static void sw(unsigned a, unsigned v) {
  M[a & 0xffff] = v & 0xff;
  M[(a + 1) & 0xffff] = (v >> 8) & 0xff;
}

// An ISIS file name at addr: device ("F0" when none is given, "CI", ...)
// and NAME.EXT, upper case.
static bool parse_name(unsigned addr, std::string &dev, std::string &name) {
  unsigned a = addr;
  while (M[a] == ' ') a++;
  dev = "F0";
  if (M[a] == ':') {
    dev.clear();
    dev += (char)toupper(M[a + 1]);
    dev += (char)toupper(M[a + 2]);
    if (M[a + 3] != ':') return false;
    a += 4;
  }
  name.clear();
  int n = 0;
  while (isalnum(M[a]) && n < 6) { name += (char)toupper(M[a]); a++; n++; }
  if (M[a] == '.') {
    name += '.';
    a++;
    n = 0;
    while (isalnum(M[a]) && n < 3) { name += (char)toupper(M[a]); a++; n++; }
  }
  return true;
}

// The host file an ISIS name refers to: an existing file of that name in any
// case, else the name as given.
static std::string host_path(const std::string &name) {
  DIR *d = opendir(".");
  if (d) {
    struct dirent *e;
    while ((e = readdir(d))) {
      if (strcasecmp(e->d_name, name.c_str()) == 0) {
        std::string r = e->d_name;
        closedir(d);
        return r;
      }
    }
    closedir(d);
  }
  return name;
}

// Load an absolute object module (content records only) at bias; entry is
// the start address from MODEND.
static bool load_omf(const std::string &path, unsigned bias, unsigned &entry) {
  FILE *f = fopen(host_path(path).c_str(), "rb");
  if (!f) return false;
  std::vector<unsigned char> b;
  int c;
  while ((c = fgetc(f)) != EOF) b.push_back((unsigned char)c);
  fclose(f);
  size_t p = 0;
  entry = 0;
  while (p + 3 <= b.size()) {
    unsigned type = b[p];
    unsigned len = b[p + 1] | (b[p + 2] << 8);
    if (p + 3 + len > b.size()) break;
    const unsigned char *r = &b[p + 3];
    if (type == 0x06) {  // CONTENT: segment, offset, bytes, checksum
      unsigned off = r[1] | (r[2] << 8);
      for (unsigned i = 3; i + 1 < len; i++) M[(off + bias + i - 3) & 0xffff] = r[i];
    } else if (type == 0x04) {  // MODEND: type, segment, offset
      entry = (r[2] | (r[3] << 8)) + bias;
    } else if (type == 0x0e) {  // EOF
      break;
    }
    p += 3 + len;
  }
  return true;
}

static int find_aft() {
  for (int i = 2; i < 32; i++)
    if (!afts[i].used) return i;
  return -1;
}

// ISIS system call fn with its parameter block at pb; returns to the caller
// unless the call was a LOAD that transfers control.
static void isis_call(unsigned fn, unsigned pb) {
  unsigned p[6];
  for (int i = 0; i < 6; i++) p[i] = w(pb + 2 * i);
  if (trace_isis)
    fprintf(stderr, "ISIS %u pb=%04x %04x %04x %04x %04x %04x\n", fn, p[0], p[1], p[2], p[3], p[4], p[5]);
  switch (fn) {
  case 0: {  // OPEN aftptr, file, access, echo, status
    std::string dev, name;
    int n = find_aft();
    if (!parse_name(p[1], dev, name)) { sw(p[4], 4); break; }
    if (n < 0) { sw(p[4], 3); break; }
    AFT &a = afts[n];
    a = AFT();
    a.access = p[2];
    if (dev == "CI" || dev == "TI" || dev == "VI") {
      a.used = true;
      a.console_in = true;
    } else if (dev == "CO" || dev == "TO" || dev == "VO") {
      a.used = true;
      a.console_out = true;
    } else if (dev == "BB") {
      a.used = true;
      a.bucket = true;
    } else if (dev == "LP" || dev == "L1") {
      a.used = true;
      a.path = "LP.OUT";
      a.f = fopen("LP.OUT", "ab");
    } else {
      std::string hp = host_path(name);
      const char *mode = p[2] == 1 ? "rb" : p[2] == 2 ? "wb" : "r+b";
      FILE *f = fopen(hp.c_str(), mode);
      if (!f && p[2] == 3) f = fopen(hp.c_str(), "w+b");
      if (!f) {
        sw(p[4], 13);  // no such file
        if (trace_isis) fprintf(stderr, "  open %s failed\n", name.c_str());
        break;
      }
      setvbuf(f, NULL, _IONBF, 0);
      a.used = true;
      a.f = f;
      a.path = hp;
    }
    sw(p[0], n);
    sw(p[4], 0);
    if (trace_isis) fprintf(stderr, "  open :%s:%s -> aft %d\n", dev.c_str(), name.c_str(), n);
    break;
  }
  case 1: {  // CLOSE aft, status
    unsigned n = p[0];
    if (n < 32 && n >= 2 && afts[n].used) {
      if (afts[n].f) fclose(afts[n].f);
      afts[n] = AFT();
    }
    sw(p[1], 0);
    break;
  }
  case 2: {  // DELETE file, status
    std::string dev, name;
    parse_name(p[0], dev, name);
    int r = unlink(host_path(name).c_str());
    sw(p[1], r == 0 ? 0 : 13);
    break;
  }
  case 3: {  // READ aft, buffer, count, actual, status
    unsigned n = p[0], buf = p[1], cnt = p[2];
    unsigned got = 0;
    AFT *a = n < 32 ? &afts[n] : nullptr;
    if (n == 1 || (a && a->console_in)) {
      // The console gives a line at a time; the command line comes first.
      if (cipos >= cmdline.size()) {
        char line[512];
        if (fgets(line, sizeof line, stdin)) {
          size_t L = strlen(line);
          while (L && (line[L - 1] == '\n' || line[L - 1] == '\r')) line[--L] = 0;
          cmdline = std::string(line) + "\r\n";
        } else {
          cmdline = std::string("\x1a");
        }
        cipos = 0;
      }
      while (got < cnt && cipos < cmdline.size()) {
        char ch = cmdline[cipos++];
        M[(buf + got++) & 0xffff] = ch;
        if (ch == '\n') break;
      }
    } else if (a && a->f) {
      std::vector<unsigned char> tmp(cnt);
      fseek(a->f, a->pos, SEEK_SET);
      got = (unsigned)fread(tmp.data(), 1, cnt, a->f);
      a->pos += got;
      for (unsigned i = 0; i < got; i++) M[(buf + i) & 0xffff] = tmp[i];
    }
    sw(p[3], got);
    sw(p[4], 0);
    break;
  }
  case 4: {  // WRITE aft, buffer, count, status
    unsigned n = p[0], buf = p[1], cnt = p[2];
    AFT *a = n < 32 ? &afts[n] : nullptr;
    if (n == 0 || (a && a->console_out)) {
      for (unsigned i = 0; i < cnt; i++) {
        unsigned char ch = M[(buf + i) & 0xffff];
        if (ch != '\r') fputc(ch & 0x7f, stdout);
      }
      fflush(stdout);
    } else if (a && a->f) {
      fseek(a->f, a->pos, SEEK_SET);
      for (unsigned i = 0; i < cnt; i++) fputc(M[(buf + i) & 0xffff], a->f);
      a->pos += cnt;
      fflush(a->f);
    }
    sw(p[3], 0);
    break;
  }
  case 5: {  // SEEK aft, mode, blockptr, byteptr, status
    unsigned n = p[0], mode = p[1];
    AFT *a = n < 32 ? &afts[n] : nullptr;
    if (!a || !a->f) { sw(p[4], 0); break; }
    long cur = a->pos;
    long amt = (long)w(p[2]) * 128 + w(p[3]);
    fseek(a->f, 0, SEEK_END);
    long end = ftell(a->f);
    long np = cur;
    switch (mode) {
    case 0: sw(p[2], (unsigned)(cur / 128)); sw(p[3], (unsigned)(cur % 128)); break;
    case 1: np = cur - amt; if (np < 0) np = 0; break;
    case 2: np = amt; break;
    case 3: np = cur + amt; break;
    case 4: np = end; break;
    }
    if (np > end) {
      if (a->access == 1) np = end;
      else { fseek(a->f, 0, SEEK_END); for (long i = end; i < np; i++) fputc(0, a->f); }
    }
    a->pos = np;
    fflush(a->f);
    sw(p[4], 0);
    break;
  }
  case 6: {  // LOAD file, bias, retsw, entry, status
    std::string dev, name;
    parse_name(p[0], dev, name);
    unsigned entry;
    if (!load_omf(name, p[1], entry)) {
      sw(p[4], 13);
      fprintf(stderr, "isis: LOAD %s failed\n", name.c_str());
      break;
    }
    if (p[2] == 0) {
      sw(p[4], 0);
      sw(p[3], entry);
    } else {  // run it: the ISIS call does not return
      cpu.pop_word();
      cpu.regs.PC.set_pair16(entry);
      return;
    }
    break;
  }
  case 7: {  // RENAME old, new, status
    std::string d1, n1, d2, n2;
    parse_name(p[0], d1, n1);
    parse_name(p[1], d2, n2);
    int r = rename(host_path(n1).c_str(), n2.c_str());
    sw(p[2], r == 0 ? 0 : 13);
    break;
  }
  case 8: sw(p[2], 0); break;  // CONSOL
  case 9: fflush(stdout); exit(0);  // EXIT
  case 10: sw(p[3], 0); break;  // ATTRIB
  case 11: cipos = 0; sw(p[1], 0); break;  // RESCAN aft, status
  case 12:  // ERROR errnum, status
    fprintf(stdout, "\nERROR %u USER PC %04X\n", p[0], w(cpu.regs.SP.get_pair16()));
    sw(p[1], 0);
    break;
  case 13: {  // WHOCON aft, buffer, status
    const char *s = p[0] == 1 ? ":CI: " : ":CO: ";
    for (int i = 0; s[i]; i++) M[(p[1] + i) & 0xffff] = s[i];
    sw(p[2], 0);
    break;
  }
  case 14: {  // SPATH file, info, status
    std::string dev, name;
    parse_name(p[0], dev, name);
    unsigned info = p[1];
    for (int i = 0; i < 12; i++) M[(info + i) & 0xffff] = 0;
    int devno;
    if (dev[0] == 'F') devno = dev[1] - '0';
    else if (dev == "CI") devno = 27;
    else if (dev == "CO") devno = 28;
    else if (dev == "BB") devno = 26;
    else if (dev == "LP") devno = 24;
    else devno = 10;
    M[info & 0xffff] = devno;
    std::string base = name, ext;
    size_t dot = name.find('.');
    if (dot != std::string::npos) { base = name.substr(0, dot); ext = name.substr(dot + 1); }
    for (size_t i = 0; i < base.size() && i < 6; i++) M[(info + 1 + i) & 0xffff] = base[i];
    for (size_t i = 0; i < ext.size() && i < 3; i++) M[(info + 7 + i) & 0xffff] = ext[i];
    M[(info + 10) & 0xffff] = devno <= 9 ? 3 : (devno == 27 ? 0 : 1);
    M[(info + 11) & 0xffff] = devno <= 9 ? 4 : 0;
    sw(p[2], 0);
    break;
  }
  default:
    fprintf(stderr, "isis: unknown ISIS call %u\n", fn);
    sw(p[5], 0);
  }
  unsigned ret = cpu.pop_word();
  cpu.regs.PC.set_pair16(ret);
}

int main(int argc, char **argv) {
  if (getenv("ISIS_TRACE")) trace_isis = true;
  if (argc < 2) {
    fprintf(stderr, "usage: isis PROGRAM [arguments...]\n");
    return 2;
  }
  unsigned memtop = 0xF6EF;
  if (getenv("MEMTOP")) memtop = (unsigned)strtol(getenv("MEMTOP"), 0, 16) & 0xffff;
  cpu.set_cpu_mode(qkz80::MODE_8080);
  M = cpu.get_mem();
  memset(M, 0, 65536);
  afts[0].used = true;
  afts[0].console_out = true;
  afts[1].used = true;
  afts[1].console_in = true;
  cmdline = argv[1];
  for (int i = 2; i < argc; i++) {
    cmdline += " ";
    cmdline += argv[i];
  }
  cmdline += "\r\n";
  cipos = strlen(argv[1]);  // the program reads its arguments after its name
  // Parse the program's name with the same rules as an ISIS call's.
  std::string pn = argv[1], dev, prog;
  if (pn.size() > 60) pn.resize(60);
  for (size_t i = 0; i < pn.size(); i++) M[0x3000 + i] = pn[i];
  M[0x3000 + pn.size()] = 0;
  parse_name(0x3000, dev, prog);
  memset(&M[0x3000], 0, 64);
  unsigned entry;
  if (!load_omf(prog, 0, entry)) {
    fprintf(stderr, "isis: cannot load %s\n", prog.c_str());
    return 2;
  }
  cpu.regs.PC.set_pair16(entry);
  cpu.regs.SP.set_pair16(0xF6F0);
  M[0xF6F0] = 0;
  M[0xF6F1] = 0;
  long long n = 0;
  for (;;) {
    unsigned pc = cpu.regs.PC.get_pair16();
    if (pc == 0x40) {  // the ISIS entry point
      isis_call(cpu.get_reg8(qkz80::reg_C), cpu.get_reg16(qkz80::regp_DE));
      continue;
    }
    if (pc == 0) {
      fflush(stdout);
      return 0;
    }
    if (pc >= 0xF800 && pc < 0xF820) {  // monitor entry points
      unsigned off = pc - 0xF800;
      if (off == 0x1B) {  // MEMCK
        cpu.set_reg8(memtop & 0xff, qkz80::reg_A);
        cpu.set_reg8(memtop >> 8, qkz80::reg_B);
      } else if (off == 0x09 || off == 0x0F) {  // CO, LO
        fputc(cpu.get_reg8(qkz80::reg_C) & 0x7f, stdout);
      } else if (off == 0x12 || off == 0x15) {  // CSTS, IOCHK
        cpu.set_reg8(0, qkz80::reg_A);
      } else {
        fprintf(stderr, "isis: monitor call %04x\n", pc);
      }
      unsigned ret = cpu.pop_word();
      cpu.regs.PC.set_pair16(ret);
      continue;
    }
    cpu.execute();
    if (++n > 4000000000LL) {
      fprintf(stderr, "isis: instruction limit\n");
      return 3;
    }
  }
}
