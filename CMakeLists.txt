cmake_minimum_required (VERSION 3.10)

# Activate Hot Reload for MSVC-Compiler, as long as it is supported.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project (cnn VERSION "1.0.0" DESCRIPTION "Neuronal Network in C trained on MNIST dataset" LANGUAGES C)

add_executable (cnn "main.c" "source/dataset.c" "source/neuronal_network.c")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET cnn PROPERTY C_STANDARD 17)
endif()

set(PROJECT_DATA_DIR "${PROJECT_SOURCE_DIR}/datasets/")
configure_file("main.h.in" "${PROJECT_SOURCE_DIR}/headers/cmake_variables.h")

target_include_directories(cnn PUBLIC "${PROJECT_SOURCE_DIR}")
