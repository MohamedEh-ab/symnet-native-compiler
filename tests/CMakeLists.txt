# Test registration: C++ unit tests + benchmark, a single-header sanity build, Python bindings tests.

add_executable(unit_tests cpp/test_engine.cpp)
target_link_libraries(unit_tests PRIVATE SymNet::native_cpp)
add_test(NAME symnet_unit_tests COMMAND unit_tests --quick)

add_executable(finite_field_test cpp/finite_field_test.cpp)
target_link_libraries(finite_field_test PRIVATE SymNet::native_cpp)
add_test(NAME SymNet.FiniteField COMMAND finite_field_test)

add_executable(engine_integration_test cpp/engine_integration_test.cpp)
target_link_libraries(engine_integration_test PRIVATE SymNet::native_cpp)
add_test(NAME SymNet.EngineBridge COMMAND engine_integration_test)

# The amalgamated single file must keep building and passing (guards against a stale copy).
add_executable(single_header_check ${PROJECT_SOURCE_DIR}/single_header/symnet_native_single.cpp)
target_link_libraries(single_header_check PRIVATE symnet_native_flags)
target_compile_features(single_header_check PRIVATE cxx_std_17)
add_test(NAME symnet_single_header COMMAND single_header_check --quick)

# Python bindings (only when the extension module was built).
if(TARGET symnet_native)
  add_test(NAME symnet_python_bindings
           COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/python/test_bindings.py")
  set_tests_properties(symnet_python_bindings PROPERTIES
      ENVIRONMENT "PYTHONPATH=$<TARGET_FILE_DIR:symnet_native>")
endif()
