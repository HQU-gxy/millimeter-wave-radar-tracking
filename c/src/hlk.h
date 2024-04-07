#ifndef E1A63B77_CF24_4CF0_AC38_40EC28E0338A
#define E1A63B77_CF24_4CF0_AC38_40EC28E0338A

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#include <cstring>
#else
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct hlk_target {
  /*
   * @brief enable or disable the target
   *
   * 0: disable
   * 1: enable
   */
  uint8_t en;
  int16_t x;
  int16_t y;
  int16_t speed;
  uint16_t resolution;
};

typedef struct hlk_target hlk_target_t;

struct hlk_result {
  hlk_target_t targets[3];
};

typedef struct hlk_result hlk_result_t;

/**
 * @brief convert a msb signed short to a int16_t
 * @param value the msb signed short
 * @return int16_t the converted value
 */
int16_t hlk_msb_signed_short(uint16_t value);

/**
 * @brief unmarshal a target from a byte array
 *
 * @param [in] data the byte array
 * @param [in] size the size of the byte array
 * @param [out] target the target to unmarshal
 * @return int 0 on success, otherwise failure
 */
int hlk_unmarshal_target(const uint8_t *data, size_t size,
                         hlk_target_t *target);

/**
 * @brief unmarshal a hlk_result from a byte array
 * @param [in] data the byte array
 * @param [in] size the size of the byte array
 * @param [in] result the result to unmarshal
 * @return int 0 on success, otherwise failure
 */
int hlk_unmarshal_result(const uint8_t *data, size_t size,
                         hlk_result_t *result);

#ifdef __cplusplus
}
#endif
#endif /* E1A63B77_CF24_4CF0_AC38_40EC28E0338A */
