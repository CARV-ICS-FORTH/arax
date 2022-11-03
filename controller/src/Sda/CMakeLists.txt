enable_backend(OFF)

if(ENABLE_Sda)
set(XILINX_SDACCEL "" CACHE STRING "Xilinx SDAaccel installation directory")

# Old version references a xcl.cpp not found in current version
aux_source_directory(./ sda_sources)
add_library(sda OBJECT ${sda_sources})
target_link_libraries(sda PRIVATE xilinxopencl PRIVATE lmx6.0)

vc_link_target(sda)
endif(ENABLE_Sda)
