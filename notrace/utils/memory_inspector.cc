#include "utils/memory_inspector.h"
#include <optional>

namespace notrace {

StringStore& strongStore = StringStore::getInstance();

void MemoryRegion::print() const {
  printf("0x%llx-0x%llx %s %llx %x:%x %llu %s\n", start_addr, end_addr,
         strongStore.getStringFromId(permsId).c_str(), offset, dev_major,
         dev_minor, inode, strongStore.getStringFromId(pathnameId).c_str());
}

std::optional<MemoryRegion> find_address_in_maps(const void* target_address) {
  const unsigned long long target_addr =
      reinterpret_cast<unsigned long long>(target_address);
  auto maps = read_all_maps();
  for (const auto& region : maps) {
    if (target_addr >= region.start_addr && target_addr < region.end_addr) {
      return region;
    }
  }
  return std::nullopt;
}

std::optional<MemoryRegion> parse_maps_line(const std::string& line) {
  MemoryRegion region;
  char pathname_buf[4096] = {0};
  char perms[5];
  int fields =
      sscanf(line.c_str(), "%llx-%llx %4s %llx %x:%x %llu %4095[^\n]",
             &region.start_addr, &region.end_addr, perms, &region.offset,
             &region.dev_major, &region.dev_minor, &region.inode, pathname_buf);
  region.permsId = strongStore.getStringId(std::string(perms));
  if (fields >= 7) {
    const std::string pathname = (fields == 8) ? std::string(pathname_buf) : "";
    region.pathnameId = strongStore.getStringId(pathname);
    return region;
  }
  return std::nullopt;
}

std::vector<MemoryRegion> read_all_maps() {
  std::vector<MemoryRegion> regions;
  FILE* maps_file = fopen("/proc/self/maps", "r");
  if (!maps_file) {
    perror("Failed to open /proc/self/maps");
    return regions;
  }

  char line[8192];
  while (fgets(line, sizeof(line), maps_file)) {
    auto region_opt = parse_maps_line(std::string(line));
    if (region_opt) {
      regions.push_back(*region_opt);
    }
  }

  fclose(maps_file);
  return regions;
}

}  // namespace notrace
