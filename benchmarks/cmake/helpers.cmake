function(add_flags target)
    target_compile_definitions(${target} PRIVATE ${ADDITIONAL_DEFINITIONS})
    target_compile_options(${target} PRIVATE ${ADDITIONAL_COMPILE_OPTIONS})
    target_link_options(${target} PRIVATE ${ADDITIONAL_LINK_OPTIONS})
endfunction()

function(add_binary target source include)
    add_executable(${target} ${source})
    add_flags(${target})

    # NOTE: should be benchmark::benchmark but doesn't work
    target_link_libraries(${target} PRIVATE base ${ADDITIONAL_LIBRARIES})
    target_include_directories(${target} PRIVATE ${include})
endfunction()
