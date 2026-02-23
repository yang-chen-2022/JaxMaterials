include(GoogleTest)
add_subdirectory(tests)

# Lippmann Schwinger solver library
add_library(lippmannschwinger SHARED)

target_sources(lippmannschwinger
    PRIVATE
        FILE_SET internalOnlyHeaders
        TYPE HEADERS
        FILES           
    
    PRIVATE        
        common.cc
        derivatives.cu        
        fourier_solve.cu
        profile.cu
        lippmann_schwinger.cu
        
    PUBLIC
        FILE_SET HEADERS
        FILES            
            common.hh
            derivatives.hh                        
            fourier_solve.hh
            profile.hh
            lippmann_schwinger.hh
)

# Link against google test
target_link_libraries(lippmannschwinger
    PRIVATE
        CUDA::cufft
        GTest::gtest_main         
)

set_target_properties(lippmannschwinger PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ../lib
    POSITION_INDEPENDENT_CODE ON
)

# Driver executable
add_executable(driver)
target_link_libraries(driver 
    PRIVATE
        CUDA::cufft
        lippmannschwinger
)
target_sources(driver
    PRIVATE
        driver.cu
)
set_target_properties(driver PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# Test executable
add_executable(test)
target_sources(test
    PRIVATE
        test.cu
)
target_link_libraries(test
    PRIVATE
        CUDA::cufft
        GTest::gtest_main         
        lippmannschwinger
)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# Find google tests
gtest_discover_tests(test)