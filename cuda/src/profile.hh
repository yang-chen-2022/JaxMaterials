#ifndef PROFILE_HH
#define PROFILE_HH PROFILE_HH
#include "common.hh"
#include "derivatives.hh"
#include <chrono>

/** @brief Profile derivative computation
 *
 * Measure time spent in derivative computations
 */
void profile_derivatives();
#endif // PROFILE_HH