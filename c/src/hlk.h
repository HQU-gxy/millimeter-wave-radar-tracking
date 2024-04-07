#ifndef E1A63B77_CF24_4CF0_AC38_40EC28E0338A
#define E1A63B77_CF24_4CF0_AC38_40EC28E0338A

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>
#include <cstring>
#else
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief boolean true
 */
#define HLK_OK 1
/**
 * @brief boolean false
 */
#define HLK_ERR 0

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
 * @brief unmarshal a target from a byte array
 *
 * @param [in] data the byte array
 * @param [in] size the size of the byte array
 * @param [out] target the target to unmarshal
 * @return int HLK_OK on success, otherwise HLK_ERR
 */
int hlk_unmarshal_target(const uint8_t *data, size_t size,
                         hlk_target_t *target);

/**
 * @brief unmarshal a hlk_result from a byte array
 * @param [in] data the byte array
 * @param [in] size the size of the byte array
 * @param [in] result the result to unmarshal
 * @return int HLK_OK on success, otherwise HLK_ERR
 */
int hlk_unmarshal_result(const uint8_t *data, size_t size,
                         hlk_result_t *result);

#ifdef __cplusplus
}
#endif
#endif /* E1A63B77_CF24_4CF0_AC38_40EC28E0338A */
