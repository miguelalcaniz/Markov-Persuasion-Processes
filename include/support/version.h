/* #############################################
 *             This file is part of
 *                    ZERO
 *
 *             Copyright (c) 2020
 *     Released under the Creative Commons
 *         CC BY-NC-SA 4.0 License
 *
 *              Find out more at
 *        https://github.com/ds4dm/ZERO
 * #############################################*/


#include <string>
#define ZERO_VERSION "1.3.0"
#define ZERO_VERSION_MAJOR "1"
#define ZERO_VERSION_MINOR "3"
#define ZERO_VERSION_PATCH "0"


/**
 * @brief This class provides an helper for the version of ZERO. It should be automatically
 * configured by CMake.
 */
class ZEROVersion {
public:
  /**
   * @brief A constructor filling the parameters with the version of ZERO
   * @param major Output for the major version
   * @param minor Output for the minor version
   * @param patch Output for the patch version
   */
  ZEROVersion(std::string &major, std::string &minor, std::string &patch) {
	 major = ZERO_VERSION_MAJOR;
	 minor = ZERO_VERSION_MINOR;
	 patch = ZERO_VERSION_PATCH;
  }
  /**
   * @brief Provides the ZERO version as a single string
   * @param version The output version string
   */
  explicit ZEROVersion(std::string &version) { version = ZERO_VERSION; }
};
