cmake_minimum_required(VERSION 3.11)
cmake_policy(SET CMP0072 NEW)

file(GLOB TD_DIRECTORIES "Ex*")

foreach(TD ${TD_DIRECTORIES})
    add_subdirectory(${TD})
	# get_filename_component(DIRN ${TD} NAME)
    # add_custom_target(${DIRN} COMMAND cd ${CMAKE_CURRENT_BINARY_DIR}/${DIRN} COMMAND make
    #             WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/${DIRN})
endforeach()

