cmake_minimum_required(VERSION 3.16)
project(Gravity4ChessMinimax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

function(configure_engine_target target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    target_compile_options(${target} PRIVATE
      -O3
      -march=native
      -flto
      -funroll-loops
      -DNDEBUG
      -Wall
      -Wextra
      -Wpedantic
    )
    target_link_options(${target} PRIVATE -flto)
  elseif(MSVC)
    target_compile_options(${target} PRIVATE /O2 /DNDEBUG /W4)
  endif()
endfunction()

add_executable(gravity4_play play_cpp_opti.cpp)
configure_engine_target(gravity4_play)

add_executable(gravity4_play_legacy play_cpp_opti_legacy.cpp)
configure_engine_target(gravity4_play_legacy)

add_executable(gravity4_benchmark benchmark.cpp)
configure_engine_target(gravity4_benchmark)

add_executable(gravity4_tests engine_tests.cpp)
configure_engine_target(gravity4_tests)

enable_testing()
add_test(NAME gravity4_engine_tests COMMAND gravity4_tests)
