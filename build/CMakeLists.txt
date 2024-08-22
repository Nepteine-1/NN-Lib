cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
# Very important, otherwise your executable will be considered as a shared library by your OS and not as an executable
set(CMAKE_CXX_FLAGS "-no-pie")
set(PROJECT_NAME "$ENV{EXE_TITLE}")
project(${PROJECT_NAME})
	
# Define macro for c++ preprocessor directives
# Comment the following line to disable testing process
# add_compile_definitions(ENABLE_TEST)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DEBUG)
endif()

set(PERMISSIONS OWNER_READ 
    OWNER_EXECUTE
    OWNER_READ
    OWNER_WRITE
    GROUP_EXECUTE 
    WORLD_EXECUTE
)

set(CMAKE_SHARED_LIBRARY_PREFIX "")
set(CMAKE_STATIC_LIBRARY_PREFIX "")

include_directories(${PROJECT_SOURCE_DIR}/../dev/include)

if("$ENV{TARGET}" STREQUAL "" OR "$ENV{TARGET}" STREQUAL "dev")
	file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/../dev/src/*.cpp)
	file(GLOB_RECURSE HDRS ${PROJECT_SOURCE_DIR}/../dev/include/*.hpp)

	add_executable(${PROJECT_NAME}
    ${SRCS}
    ${HDRS})
    
	target_link_libraries(${PROJECT_NAME}
		sfml-graphics
		sfml-window
		sfml-system 
		sfml-audio
		sfml-network
		pthread
	)
	
	install(FILES ${PROJECT_NAME}  DESTINATION ${PROJECT_SOURCE_DIR}/.. PERMISSIONS ${PERMISSIONS} )
	
elseif("$ENV{TARGET}" STREQUAL "shared")
	set(PROJECT_NAME "$ENV{LIB_NAME}")
	
	file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/../dev/src/$ENV{LIB_NAME}/*.cpp)
	file(GLOB_RECURSE HDRS ${PROJECT_SOURCE_DIR}/../dev/include/$ENV{LIB_NAME}/*.hpp)

	add_library(${PROJECT_NAME} SHARED
    	${SRCS}
    	${HDRS})
    
elseif("$ENV{TARGET}" STREQUAL "static")
	set(PROJECT_NAME "$ENV{LIB_NAME}")

	file(GLOB_RECURSE SRCS ${PROJECT_SOURCE_DIR}/../dev/src/$ENV{LIB_NAME}/*.cpp)
	file(GLOB_RECURSE HDRS ${PROJECT_SOURCE_DIR}/../dev/include/$ENV{LIB_NAME}/*.hpp)
	
    add_library(${PROJECT_NAME}
    	${SRCS}
    	${HDRS})
    	
endif()
