cmake_minimum_required(VERSION 3.25)
project(llm_scheduler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(-Wall -Wextra)

set(SANITIZE "" CACHE STRING "Sanitizers to build with, e.g. address,undefined or thread")
if(SANITIZE)
    add_compile_options(-fsanitize=${SANITIZE} -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=${SANITIZE})
endif()

add_library(llm_scheduler STATIC
    src/block_pool.cpp
)
target_include_directories(llm_scheduler PUBLIC src)

enable_testing()

function(add_scheduler_test name)
    add_executable(${name} tests/${name}.cpp)
    target_link_libraries(${name} PRIVATE llm_scheduler)
    add_test(NAME ${name} COMMAND ${name})
endfunction()

add_scheduler_test(block_pool_test)
