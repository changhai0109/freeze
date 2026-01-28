#ifndef NOTRACE_UTILS_MEMORY_INSPECTOR_H_
#define NOTRACE_UTILS_MEMORY_INSPECTOR_H_

#include <cstdio>
#include <optional>
#include <string>
#include <vector>
#include "utils/string_store.h"

namespace notrace {

#pragma pack(push, 1)
struct MemoryRegion {
  unsigned long long start_addr;
  unsigned long long end_addr;
  StringId permsId;
  unsigned long long offset;
  int dev_major;
  int dev_minor;
  unsigned long long inode;
  StringId pathnameId;

  // Helper to print a region
  void print() const;
};
#pragma pack(pop)

std::optional<MemoryRegion> parse_maps_line(const std::string& line);
std::vector<MemoryRegion> read_all_maps();
std::optional<MemoryRegion> find_address_in_maps(const void* target_address);

}  // namespace notrace
#endif
