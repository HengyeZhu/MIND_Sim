#include "utils/dynamic_library.hpp"

#include "coreneuron/mechanism/membfunc.hpp"

#include <dlfcn.h>

#include <stdexcept>
#include <utility>

namespace mind_sim::utils {
namespace {

void promote_host_symbols_to_global() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&coreneuron::nrn_get_mechtype), &info) == 0 ||
        info.dli_fname == nullptr) {
        throw std::runtime_error("failed to resolve MIND_Sim extension for dynamic module loading");
    }

    dlerror();
    void* handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL | RTLD_NOLOAD);
    if (handle == nullptr) {
        dlerror();
        handle = dlopen(info.dli_fname, RTLD_NOW | RTLD_GLOBAL);
    }
    if (handle == nullptr) {
        const char* error = dlerror();
        throw std::runtime_error(
            std::string("failed to promote MIND_Sim extension symbols for dynamic module loading: ") +
            (error != nullptr ? error : "unknown dlopen error"));
    }
}

}  // namespace

DynamicLibrary::DynamicLibrary(std::string path)
    : path_(std::move(path)) {
    promote_host_symbols_to_global();
    handle_ = dlopen(path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle_) {
        throw std::runtime_error("failed to load generated module '" + path_ + "': " + dlerror());
    }
}

DynamicLibrary::~DynamicLibrary() {
    close();
}

DynamicLibrary::DynamicLibrary(DynamicLibrary&& other) noexcept
    : path_(std::move(other.path_)), handle_(other.handle_) {
    other.handle_ = nullptr;
}

DynamicLibrary& DynamicLibrary::operator=(DynamicLibrary&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        handle_ = other.handle_;
        other.handle_ = nullptr;
    }
    return *this;
}

void* DynamicLibrary::symbol(const char* name) const {
    dlerror();
    void* ptr = dlsym(handle_, name);
    const char* error = dlerror();
    if (error || !ptr) {
        throw std::runtime_error("generated module '" + path_ + "' is missing symbol '" +
                                 std::string(name) + "'");
    }
    return ptr;
}

const std::string& DynamicLibrary::path() const noexcept {
    return path_;
}

void DynamicLibrary::close() noexcept {
    if (handle_) {
        dlclose(handle_);
        handle_ = nullptr;
    }
}

std::shared_ptr<DynamicLibrary> load_dynamic_library(const std::string& path) {
    return std::make_shared<DynamicLibrary>(path);
}

}  // namespace mind_sim::utils
