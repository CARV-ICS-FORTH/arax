/**
 * @file
 * Persistent configuration utility functions.
 *
 * The configuration file resides in ~/.vinetalk.
 *
 * Configuration file format:
 * Each line(terminated with a newline character \n) contains a single
 * key value pair in the following format:
 *
 * KEY VALUE\n
 *
 * The key value can have any pritable character except white space,
 * it should not be larger than 32 characters.
 * The value begins after KEY and any whites-pace after it.
 * Value ends at the first newline character.
 *
 */
#ifndef VINEYARD_CONFIG_HEADER
#define VINEYARD_CONFIG_HEADER
#include <stddef.h>

/**
 * Get value corresponding to \c key as a string
 *
 * Will search the ~/.vinetalk file for a key/value pair matching the \c key.
 * If the value is not found 0 will be returned.
 * If during the search any error occurs, 0 will be returned and
 * a message will be printed on stderr.
 * \note This is a very slow function, use it only during initialization.
 *
 * @param key c style string string, with the key of interest.
 * @param value pointer to allocated array of size \c value_size.
 * @param value_size Size of \c value array, in bytes.
 * @return Zero on failure.
 */
int util_config_get_str(const char *key, char *value, size_t value_size);

/**
 * Get value corresponding to \c key as a boolean (0,1)
 *
 * Will search the ~/.vinetalk file for a key/value pair matching the \c key.
 * If the value is not found 0 will be returned.
 * If during the search any error occurs, 0 will be returned and
 * a message will be printed on stderr.
 * If the value is found and the value equals to 1 then \c *val will
 * be assigned 1, otherwise *val will be set to 0.
 * \note This is a very slow function, use it only during initialization.
 *
 * @param key c style string, with the key of interest.
 * @param value pointer to allocated array of size \c value_size.
 * @param def_val Default value in case the key is not found.
 * @return Zero on failure.
 */
int util_config_get_bool(const char *key, int *val,int def_val);

/**
 * Get value corresponding to \c key as an integer
 *
 * Will search the ~/.vinetalk file for a key/value pair matching the \c key.
 * If during the search any error occurs, 0 will be returned and
 * a message will be printed on stderr.
 * If the value is found it will be converted to an integer using atoi() and
 * the result will be assigned to \c val.
 * If no value is found \c val will be assigned \c def_val and 0 will
 * be returned.
 * \note This is a very slow function, use it only during initialization.
 *
 * @param key c style string string, with the key of interest.
 * @param value pointer to allocated array of size \c value_size.
 * @param def_val Default value in case the key is not found.
 * @return Zero on failure.
 */
int util_config_get_int(const char *key, int *val,int def_val);

/**
 * Get value corresponding to \c key as a long
 *
 * Will search the ~/.vinetalk file for a key/value pair matching the \c key.
 * If during the search any error occurs, 0 will be returned and
 * a message will be printed on stderr.
 * If the value is found it will be converted to a long using strtol() and
 * the result will be assigned to \c val.
 * If no value is found \c val will be assigned \c def_val and 0 will
 * be returned.
 * \note This is a very slow function, use it only during initialization.
 *
 * @param key c style string string, with the key of interest.
 * @param value pointer to allocated array of size \c value_size.
 * @param def_val Default value in case the key is not found.
 * @return Zero on failure.
 */
int util_config_get_long(const char *key, long *val,long def_val);

/**
 * Get value corresponding to \c key as a size_t
 *
 * Value is retrieved with util_config_get_long().
 * If acquired value value is in [SIZE_MAX,0] it is assigned to
 * \c val and 1 is returned.
 *
 * Otherwise 0 is returned and \c val is assigned \c def_val.
 *
 * \note This function only works for numbers <= 2^63.
 * @param key c style string string, with the key of interest.
 * @param value pointer to allocated array of size \c value_size.
 * @param def_val Default value in case value was not found/appropriate.
 * @return Zero on failure.
 */
int util_config_get_size(const char *key, size_t *val,size_t def_val);

#endif /* ifndef VINEYARD_CONFIG_HEADER */
