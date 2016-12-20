#ifndef VINE_ACCEL_TYPES_HEADER
#define VINE_ACCEL_TYPES_HEADER

#ifdef __cplusplus
extern "C" {
#endif /* ifdef __cplusplus */

/**
 * Accelerator type enumeration.
 * NOTE: If updated update types_map variable in vine_accel_types.c
 */
typedef enum vine_accel_type {
	ANY       = 0,   /**< Let Scheduler Decide                 */
	GPU       = 1,   /**< Run on GPU with CUDA                 */
	GPU_SOFT  = 2,   /**< Run on CPU with software CUDA        */
	CPU       = 3,   /**< Run Native x86 code                  */
	SDA       = 4,   /**< Xilinx SDAaccel                      */
	NANO_ARM  = 5,   /**< ARM accelerator core from NanoStream */
	NANO_CORE = 6,   /**< NanoStreams FPGA accelerator         */
	VINE_ACCEL_TYPES /** End Marker                            */
} vine_accel_type_e;

/**
 * Convert a vine_accel_type_e value to a human readable string.
 * If \c type not a valid vine_accel_type_e value NULL is returned.
 * NOTE: This function should not be used in critical paths!
 *
 * @return A cahracter representation for the given \c type,NULL on error.
 */
const char * vine_accel_type_to_str(vine_accel_type_e type);

/**
 * Convert a string to the matching vine_accel_type_e value.
 * \c type will be compared ignoring capitalization with the string in
 * types_map variable in vine_accel_types.c.
 *
 * NOTE: This function should not be used in critical paths!
 *
 * @return A value from vine_accel_type_e, if no match is found returns
 * VINE_ACCEL_TYPES
 */
vine_accel_type_e vine_accel_type_from_str(const char * type);

#ifdef __cplusplus
}
#endif /* ifdef __cplusplus */

#endif
