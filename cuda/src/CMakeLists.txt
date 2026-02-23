include(GoogleTest)
add_subdirectory(tests)

# Lippmann Schwinger solver library
add_library(LippmannSchwingerLib)

target_sources(LippmannSchwingerLib
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
target_link_libraries(LippmannSchwingerLib
    PRIVATE
        CUDA::cufft
        GTest::gtest_main         
)

# Driver executable
add_executable(driver)
target_link_libraries(driver 
    PRIVATE
        LippmannSchwingerLib
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
        GTest::gtest_main         
        LippmannSchwingerLib
)
set_target_properties(test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ../bin)

# Find google tests
gtest_discover_tests(test)