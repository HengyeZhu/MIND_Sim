#include "io/result_hdf5.hpp"

#include <algorithm>
#include <cstddef>
#include <filesystem>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

namespace mind_sim::io {

namespace {

struct H5File {
    hid_t id{-1};
    explicit H5File(const std::string& path)
        : id(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)) {
        if (id < 0) {
            throw std::runtime_error("failed to create HDF5 output: " + path);
        }
    }
    H5File(const H5File&) = delete;
    H5File& operator=(const H5File&) = delete;
    ~H5File() {
        if (id >= 0) {
            H5Fclose(id);
        }
    }
};

struct H5Space {
    hid_t id{-1};
    explicit H5Space(const std::vector<hsize_t>& dims)
        : id(H5Screate_simple(static_cast<int>(dims.size()), dims.data(), nullptr)) {
        if (id < 0) {
            throw std::runtime_error("failed to create HDF5 dataspace");
        }
    }
    H5Space(const H5Space&) = delete;
    H5Space& operator=(const H5Space&) = delete;
    ~H5Space() {
        if (id >= 0) {
            H5Sclose(id);
        }
    }
};

struct H5Dataset {
    hid_t id{-1};
    H5Dataset(hid_t file, const std::string& name, hid_t type, const H5Space& space)
        : id(H5Dcreate2(file, name.c_str(), type, space.id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) {
        if (id < 0) {
            throw std::runtime_error("failed to create HDF5 dataset: " + name);
        }
    }
    H5Dataset(const H5Dataset&) = delete;
    H5Dataset& operator=(const H5Dataset&) = delete;
    ~H5Dataset() {
        if (id >= 0) {
            H5Dclose(id);
        }
    }
};

struct H5Type {
    hid_t id{-1};
    explicit H5Type(hid_t base): id(H5Tcopy(base)) {
        if (id < 0) {
            throw std::runtime_error("failed to create HDF5 type");
        }
    }
    H5Type(const H5Type&) = delete;
    H5Type& operator=(const H5Type&) = delete;
    ~H5Type() {
        if (id >= 0) {
            H5Tclose(id);
        }
    }
};

void ensure_parent_directory(const std::string& path) {
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void write_dataset(hid_t file,
                   const std::string& name,
                   hid_t type,
                   const std::vector<hsize_t>& dims,
                   const void* data) {
    H5Space space(dims);
    H5Dataset dataset(file, name, type, space);
    if (H5Dwrite(dataset.id, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
        throw std::runtime_error("failed to write HDF5 dataset: " + name);
    }
}

void write_double_dataset(hid_t file,
                          const std::string& name,
                          const std::vector<hsize_t>& dims,
                          const double* data) {
    write_dataset(file, name, H5T_NATIVE_DOUBLE, dims, data);
}

void write_float_dataset(hid_t file,
                         const std::string& name,
                         const std::vector<hsize_t>& dims,
                         const float* data) {
    write_dataset(file, name, H5T_NATIVE_FLOAT, dims, data);
}

void write_int_dataset(hid_t file,
                       const std::string& name,
                       const std::vector<hsize_t>& dims,
                       const int* data) {
    write_dataset(file, name, H5T_NATIVE_INT, dims, data);
}

void write_strings(hid_t file, const std::string& name, const std::vector<std::string>& values) {
    std::size_t width = 1;
    for (const auto& value: values) {
        width = std::max(width, value.size() + 1);
    }
    std::vector<char> bytes(values.size() * width, '\0');
    for (std::size_t row = 0; row < values.size(); ++row) {
        std::copy(values[row].begin(), values[row].end(), bytes.begin() + static_cast<std::ptrdiff_t>(row * width));
    }

    H5Type type(H5T_C_S1);
    H5Tset_size(type.id, width);
    H5Tset_strpad(type.id, H5T_STR_NULLPAD);
    write_dataset(file, name, type.id, {values.size()}, bytes.data());
}

void write_int_attribute(hid_t object, const std::string& name, int value) {
    H5Space space({1});
    const auto attr = H5Acreate2(object, name.c_str(), H5T_NATIVE_INT, space.id, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        throw std::runtime_error("failed to create HDF5 attribute: " + name);
    }
    if (H5Awrite(attr, H5T_NATIVE_INT, &value) < 0) {
        H5Aclose(attr);
        throw std::runtime_error("failed to write HDF5 attribute: " + name);
    }
    H5Aclose(attr);
}

std::size_t step_count_of(const std::vector<double>& times) {
    return times.empty() ? 0 : times.size() - 1;
}

std::size_t record_stride(const mind_sim::macro::sim::ExposureRecord& record) {
    return record.roi_indices.size() * static_cast<std::size_t>(record.exposure_count);
}

void validate_record(const mind_sim::macro::sim::ExposureRecord& record,
                     const std::vector<double>& times,
                     const std::vector<std::string>& exposure_names,
                     const std::vector<std::string>& roi_labels) {
    if (record.exposure_count <= 0 || record.roi_count <= 0 || record.roi_indices.empty()) {
        throw std::runtime_error("save_h5 requires recorded ROI exposures");
    }
    if (exposure_names.size() != static_cast<std::size_t>(record.exposure_count)) {
        throw std::runtime_error("save_h5 exposure name count does not match result");
    }
    if (roi_labels.size() != static_cast<std::size_t>(record.roi_count)) {
        throw std::runtime_error("save_h5 ROI label count does not match result");
    }
    if (record.values.size() != times.size() * record_stride(record)) {
        throw std::runtime_error("save_h5 recorded exposure array size mismatch");
    }
}

struct SortedSpikes {
    std::vector<double> time{};
    std::vector<int> gid{};
};

SortedSpikes sort_spikes(const mind_sim::micro::sim::MicroSpikeTable& spikes) {
    std::vector<std::size_t> order(spikes.time.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return spikes.time[lhs] < spikes.time[rhs];
    });

    SortedSpikes sorted;
    sorted.time.resize(order.size());
    sorted.gid.resize(order.size());
    for (std::size_t out_index = 0; out_index < order.size(); ++out_index) {
        const auto source = order[out_index];
        sorted.time[out_index] = spikes.time[source];
        sorted.gid[out_index] = spikes.gid[source];
    }
    return sorted;
}

void write_macro_result(hid_t file,
                        const std::vector<double>& times,
                        const mind_sim::macro::sim::ExposureRecord& record,
                        const std::vector<std::string>& exposure_names,
                        const std::vector<std::string>& roi_labels,
                        const std::vector<double>& timing_s,
                        const std::vector<double>& metadata) {
    const auto step_count = step_count_of(times);
    const auto recorded_roi_count = record.roi_indices.size();
    const auto exposure_count = static_cast<std::size_t>(record.exposure_count);
    const auto stride = record_stride(record);

    write_double_dataset(file,
                         "times_ms",
                         {step_count},
                         times.data() + (times.empty() ? 0 : 1));
    write_strings(file, "exposure_names", exposure_names);
    write_strings(file, "roi_labels", roi_labels);
    write_int_dataset(file,
                      "recorded_rois",
                      {recorded_roi_count},
                      record.roi_indices.data());
    write_float_dataset(file,
                        "roi_exposures",
                        {step_count, recorded_roi_count, exposure_count},
                        record.values.data() + (times.empty() ? 0 : stride));
    write_double_dataset(file, "timing_s", {timing_s.size()}, timing_s.data());
    write_double_dataset(file, "metadata", {metadata.size()}, metadata.data());
    write_int_attribute(file, "roi_count", record.roi_count);
    write_int_attribute(file, "exposure_count", record.exposure_count);
}

void write_spikes(hid_t file, const mind_sim::micro::sim::MicroSpikeTable& spikes) {
    const auto sorted = sort_spikes(spikes);
    write_double_dataset(file, "spike_times_ms", {sorted.time.size()}, sorted.time.data());
    write_int_dataset(file, "spike_gids", {sorted.gid.size()}, sorted.gid.data());
}

}  // namespace

void save_macro_result_h5(const mind_sim::macro::sim::MacroSimulationResult& result,
                          const std::string& path,
                          const std::vector<std::string>& exposure_names,
                          const std::vector<std::string>& roi_labels,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata) {
    validate_record(result.exposures, result.times, exposure_names, roi_labels);
    ensure_parent_directory(path);
    H5File file(path);
    write_macro_result(file.id,
                       result.times,
                       result.exposures,
                       exposure_names,
                       roi_labels,
                       timing_s,
                       metadata);
}

void save_cosim_result_h5(const mind_sim::cosim::SimulationResult& result,
                          const std::string& path,
                          const std::vector<std::string>& exposure_names,
                          const std::vector<std::string>& roi_labels,
                          int spike_roi,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata) {
    validate_record(result.exposures, result.times, exposure_names, roi_labels);
    if (spike_roi < 0 || static_cast<std::size_t>(spike_roi) >= result.micro_spikes_by_roi.size()) {
        throw std::runtime_error("save_h5 spike ROI index out of range");
    }
    ensure_parent_directory(path);
    H5File file(path);
    write_macro_result(file.id,
                       result.times,
                       result.exposures,
                       exposure_names,
                       roi_labels,
                       timing_s,
                       metadata);
    write_spikes(file.id, result.micro_spikes_by_roi[static_cast<std::size_t>(spike_roi)]);
}

void save_micro_spikes_h5(const mind_sim::micro::sim::MicroSpikeTable& spikes,
                          const std::string& path,
                          const std::vector<double>& timing_s,
                          const std::vector<double>& metadata) {
    ensure_parent_directory(path);
    H5File file(path);
    write_spikes(file.id, spikes);
    write_double_dataset(file.id, "timing_s", {timing_s.size()}, timing_s.data());
    write_double_dataset(file.id, "metadata", {metadata.size()}, metadata.data());
}

void save_vector_h5(const std::vector<double>& values,
                    const std::string& path,
                    const std::string& dataset_name) {
    ensure_parent_directory(path);
    H5File file(path);
    write_double_dataset(file.id, dataset_name, {values.size()}, values.data());
}

}  // namespace mind_sim::io
