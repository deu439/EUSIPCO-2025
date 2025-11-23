cmake_minimum_required(VERSION 3.5)

project(location_typeii)

add_executable(location_typeii main_blas.cpp utils.cpp type_I.cpp type_II.cpp)
find_package(X11 REQUIRED)
find_package(GSL REQUIRED)
find_package(TIFF REQUIRED)
find_package(sciplot REQUIRED)
find_package(BLAS REQUIRED)
if(BLAS_FOUND)
    message("OpenBLAS found.")
    include_directories(${BLAS_INCLUDE_DIRS})
endif(BLAS_FOUND)

# FFTW package is handled by PkgConfig
find_package(PkgConfig REQUIRED)
pkg_search_module(FFTW REQUIRED fftw3 IMPORTED_TARGET)
include_directories(PkgConfig::FFTW)

target_link_libraries(location_typeii ${X11_LIBRARIES} GSL::gsl ${BLAS_LIBRARIES} TIFF::TIFF sciplot::sciplot PkgConfig::FFTW)
install(TARGETS location_typeii RUNTIME DESTINATION bin)
