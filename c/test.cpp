#include "hlk.h"
#include <cassert>
#include <iostream>

int main() {
  const uint8_t data[] = {0xaa, 0xff, 0x03, 0x00, 0x0e, 0x03, 0xb1, 0x86,
                          0x10, 0x00, 0x40, 0x01, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                          0x00, 0x00, 0x00, 0x00, 0x55, 0xcc};
  hlk_result_t result;
  bool ok = hlk_unmarshal_result(data, sizeof(data), &result);
  for (const auto &target : result.targets) {
    std::cout << "en: " << static_cast<bool>(target.en) << std::endl;
    std::cout << "x: " << target.x << std::endl;
    std::cout << "y: " << target.y << std::endl;
    std::cout << "speed: " << target.speed << std::endl;
    std::cout << "resolution: " << target.resolution << std::endl;
  }
  assert(ok == 0);
  assert(result.targets[0].en == 0x01);
  assert(result.targets[0].x == -782);
  assert(result.targets[0].y == 1713);
  assert(result.targets[0].speed == -16);
  assert(result.targets[0].resolution == 320);
  return 0;
}
