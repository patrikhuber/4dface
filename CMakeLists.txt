project(4dface)
cmake_minimum_required(VERSION 2.8.10)
set(4dface_VERSION_MAJOR 0)
set(4dface_VERSION_MINOR 2)
set(4dface_VERSION_PATCH 0)
set(4dface_VERSION ${4dface_VERSION_MAJOR}.${4dface_VERSION_MINOR}.${4dface_VERSION_PATCH})

# Check if a supported compiler is used and add c++14 flag:
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 4.9)
		message(FATAL_ERROR "Need at least gcc 4.9 to compile.")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthread")
elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "MSVC")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 19)
		message(FATAL_ERROR "Visual Studio 2015 or newer is required.")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 3.6)
		message(FATAL_ERROR "Clang below version 3.6 will most likely not work. Please upgrade your compiler.")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pthreads")
else() # no GNU, no MSVC, no Clang
	message(WARNING "You are using an unsupported compiler. Compilation has only been tested with MSVC, GCC and Clang.")
	include(CheckCXXCompilerFlag)
	check_cxx_compiler_flag(-std=c++14 HAS_CXX14_FLAG)
	if(HAS_CXX14_FLAG)
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")
	else()
		message(FATAL_ERROR "Your compiler doesn't support the '-std=c++14' flag.")
	endif()
endif()

# Build a CPack driven installer package:
include(InstallRequiredSystemLibraries) # Includes any runtime libraries that are needed by the project for the current platform
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set(CPACK_PACKAGE_VERSION_MAJOR "${4dface_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${4dface_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${4dface_VERSION_PATCH}")
include(CPack)

# Find dependencies:
find_package(OpenCV 2.4.3 REQUIRED core imgproc highgui objdetect) # in 3.0: require videoio? Plus the thing Michael fixed for eos?
message(STATUS "OpenCV include dir found at ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV library dir found at ${OpenCV_LIB_DIR}")
set_target_properties(${OpenCV_LIBS} PROPERTIES MAP_IMPORTED_CONFIG_RELWITHDEBINFO RELEASE)

# Find the directory to OpenCV's face detector:
if(NOT OpenCV_haarcascades_DIR)
	# This will fail for system-packaged OpenCV
	list(GET OpenCV_INCLUDE_DIRS 0 OpenCV_first_include_dir)
	set(OpenCV_haarcascades_DIR "${OpenCV_first_include_dir}/../share/OpenCV/haarcascades")
endif()
if(NOT EXISTS "${OpenCV_haarcascades_DIR}/haarcascade_frontalface_alt2.xml")
	message(FATAL_ERROR "Could not find OpenCV's face detector haarcascade_frontalface_alt2.xml in ${OpenCV_haarcascades_DIR}. Please set OpenCV_haarcascades_DIR to the directory with that file.")
endif()

if(MSVC)
	# The standard find_package for boost on Win finds the dynamic libs, so for dynamic linking to boost we need to #define:
	add_definitions(-DBOOST_ALL_NO_LIB) # Don't use the automatic library linking by boost with VS (#pragma ...). Instead, we specify everything here in cmake.
	add_definitions(-DBOOST_ALL_DYN_LINK) # Link against the dynamic boost lib - needs to match with the version that find_package finds.
endif()
find_package(Boost 1.48.0 COMPONENTS system filesystem program_options REQUIRED)
if(Boost_FOUND)
  message(STATUS "Boost found at ${Boost_INCLUDE_DIRS}")
else(Boost_FOUND)
  message(FATAL_ERROR "Boost not found")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
find_package(Eigen3 REQUIRED)
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Eigen3 include dir found at ${EIGEN3_INCLUDE_DIR}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")

# Paths to our own includes, inside the git submodules:
set(eos_DIR "${CMAKE_SOURCE_DIR}/external/eos") # These are used later to gather the required data files
set(superviseddescent_DIR "${CMAKE_SOURCE_DIR}/external/superviseddescent")

set(eos_INCLUDE_DIR "${eos_DIR}/include")
set(superviseddescent_INCLUDE_DIR "${superviseddescent_DIR}/include")
set(cereal_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eos/3rdparty/cereal-1.1.1/include")
set(glm_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eos/3rdparty/glm")
set(nanoflann_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eos/3rdparty/nanoflann/include")
set(eigen3_nnls_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/external/eos/3rdparty/eigen3-nnls/src")

# The new model is not in the repository, download it manually for now:
if(NOT EXISTS "face_landmarks_model_rcr_68.bin")
	message(STATUS "Downloading face_landmarks_model_rcr_68.bin (84.6 MB)...")
	file(DOWNLOAD http://www.patrikhuber.ch/files/models/face_landmarks_model_rcr_68.bin ${CMAKE_BINARY_DIR}/face_landmarks_model_rcr_68.bin INACTIVITY_TIMEOUT 15 STATUS status SHOW_PROGRESS)
	list(GET status 0 DOWNLOAD_STATUS) # first element is status code, second is the error message
	list(GET status 1 DOWNLOAD_ERROR_MSG)
	if(NOT ${DOWNLOAD_STATUS} EQUAL 0)
		message(FATAL_ERROR "Error downloading the file: ${DOWNLOAD_STATUS}; ${DOWNLOAD_ERROR_MSG}. Please delete the file if it has been created and re-run cmake or download the file manually and put it into the root of the build directory.")
	else()
		message(STATUS "face_landmarks_model_rcr_68.bin successfully downloaded.")
	endif()
endif()

# Add header includes:
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${superviseddescent_INCLUDE_DIR})
include_directories(${eos_INCLUDE_DIR})
include_directories(${cereal_INCLUDE_DIR})
include_directories(${glm_INCLUDE_DIR})
include_directories(${nanoflann_INCLUDE_DIR})
include_directories(${eigen3_nnls_INCLUDE_DIR})

add_executable(4dface apps/4dface.cpp apps/helpers.hpp)
target_link_libraries(4dface ${OpenCV_LIBS} ${Boost_LIBRARIES})

# install targets:
install(TARGETS 4dface DESTINATION bin)
install(FILES ${eos_DIR}/share/sfm_shape_3448.bin DESTINATION share)
install(FILES ${eos_DIR}/share/ibug2did.txt DESTINATION share)
install(FILES ${eos_DIR}/share/model_contours.json DESTINATION share)
install(FILES ${eos_DIR}/share/sfm_3448_edge_topology.json DESTINATION share)
install(FILES ${eos_DIR}/share/expression_blendshapes_3448.bin DESTINATION share)
install(FILES ${CMAKE_BINARY_DIR}/face_landmarks_model_rcr_68.bin DESTINATION share)
install(FILES ${OpenCV_haarcascades_DIR}/haarcascade_frontalface_alt2.xml DESTINATION share)
