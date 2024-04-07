#include "hlk.h"

/**
 * @brief convert a msb signed short to a int16_t
 * @param value the msb signed short
 * @return int16_t the converted value
 */
static int16_t msb_signed_short(uint16_t value) {
  int8_t positive = (value & 0x8000) >= 1;
  uint16_t r = value & 0x7FFF;
  return positive ? r : -r;
}

/**
 * @brief check if all the bytes in the data are empty (0x00)
 * @param [in] data
 * @param [in] len
 * @return int 0 for false, 1 for true (all empty)
 */
static int is_all_empty(const uint8_t *data, size_t len) {
  for (int i = 0; i < len; i++) {
    if (data[i] != 0) {
      return 0;
    }
  }
  return 1;
}

/**
 * @brief check the endian of the system
 * @return int 1: little endian, 0: big endian
 */
static int calc_host_is_little_endian() {
  uint16_t x = 0x0001;
  return *((uint8_t *)&x) == 0x01;
}

/**
 * @brief check the endian of the system and cache the result
 * @return int 1: little endian, 0: big endian
 */
static int8_t is_little_endian() {
  static int8_t little_endian = -1;
  if (little_endian == -1) {
    little_endian = calc_host_is_little_endian();
  }
  return little_endian;
}

/**
 * @brief like `htons` but for little endian
 * @param x
 * @return uint16_t
 */
static uint16_t ltohs(uint16_t x) {
  return is_little_endian() ? x : ((x >> 8) | (x << 8));
}

/**
 * @brief like `ntohs` but for little endian
 * @param x
 * @return uint16_t
 */
static uint16_t htols(uint16_t x) {
  return is_little_endian() ? x : ((x >> 8) | (x << 8));
}

int hlk_unmarshal_target(const uint8_t *data, size_t len,
                         hlk_target_t *target) {
  if (len < 8) {
    return HLK_ERR;
  }
  if (is_all_empty(data, len)) {
    target->en = 0;
    target->x = 0;
    target->y = 0;
    target->speed = 0;
    target->resolution = 0;
    return HLK_OK;
  }
  int offset = 0;
  uint16_t x_ = *((uint16_t *)(data + offset));
  uint16_t x__ = ltohs(x_);
  int16_t x = msb_signed_short(x__);
  offset += 2;
  uint16_t y_ = *((uint16_t *)(data + offset));
  uint16_t y__ = ltohs(y_);
  int16_t y = msb_signed_short(y__);
  offset += 2;
  uint16_t speed_ = *((uint16_t *)(data + offset));
  uint16_t speed__ = ltohs(speed_);
  int16_t speed = msb_signed_short(speed__);
  offset += 2;
  uint16_t resolution_ = *((uint16_t *)(data + offset));
  uint16_t resolution = ltohs(resolution_);
  offset += 2;

  target->en = 1;
  target->x = x;
  target->y = y;
  target->speed = speed;
  target->resolution = resolution;
  return HLK_OK;
}

const uint8_t start_magic[] = {0xaa, 0xff, 0x03, 0x00};
const uint8_t end_magic[] = {0x55, 0xcc};

int hlk_unmarshal_result(const uint8_t *data, size_t size,
                         hlk_result_t *result) {
  size_t offset = 0;
  int ok = memcmp(data, start_magic, sizeof(start_magic));
  if (ok != 0) {
    return HLK_ERR;
  }
  offset += sizeof(start_magic);
  const int CNT = sizeof(result->targets) / sizeof(result->targets[0]);
  for (int i = 0; i < CNT; i++) {
    hlk_target_t *target = &result->targets[i];
    int ret = hlk_unmarshal_target(data + offset, 8, target);
    if (ret != HLK_OK) {
      return ret;
    }
    offset += 8;
  }
  return HLK_OK;
}
