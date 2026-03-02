
#include "common.hh"
#include "tests/test_derivatives.hh"
#include "tests/test_fourier.hh"
#include <gtest/gtest.h>

/* Main program */
int main(int argc, char *argv[])
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
  return 0;
}