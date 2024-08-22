cmake_minimum_required(VERSION 3.5)
set(CMAKE_CXX_STANDARD 11)
# Very important, otherwise your executable will be considered as a shared library by your OS and not as an executable
set(CMAKE_CXX_FLAGS "-no-pie")

set(PROJECT_NAME "Example")
project(${PROJECT_NAME})

set(PERMISSIONS OWNER_READ 
    # Set the permissions you want for your executable
    # You can choose between EXECUTE/READ/WRITE for OWNER/GROUP/WORLD
    OWNER_EXECUTE
    OWNER_READ
    OWNER_WRITE
    GROUP_EXECUTE 
    WORLD_EXECUTE
)

include_directories(${PROJECT_SOURCE_DIR}/../dev/include)
	
add_executable(${PROJECT_NAME}# Name of the executable
	main.cpp)
    	
target_link_libraries(${PROJECT_NAME}
	${PROJECT_SOURCE_DIR}/../bin/release/$ENV{LIB_NAME}.a
)
