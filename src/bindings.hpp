// bindings.hpp - declarations shared by the pybind11 translation units.
#pragma once

#include <pybind11/pybind11.h>

#include <string>

// Defined in bind_dsl.cpp / bind_finite_field.cpp; called from PYBIND11_MODULE in bindings.cpp.
void bind_dsl(pybind11::module_& root);
void bind_finite_field(pybind11::module_& root);

// Create `symnet_native.<name>` and register it in sys.modules, so both
// `symnet_native.<name>.f()` and `from symnet_native.<name> import f` work.
inline pybind11::module_ make_submodule(pybind11::module_& parent, const char* name, const char* doc) {
    pybind11::module_ sub = parent.def_submodule(name, doc);
    const std::string full = std::string("symnet_native.") + name;
    pybind11::module_::import("sys").attr("modules")[full.c_str()] = sub;
    return sub;
}
