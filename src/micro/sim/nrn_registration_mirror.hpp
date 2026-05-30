#pragma once

namespace mind_sim::micro::sim::nrn_registration_mirror {

void mechanism_registered(int type, const char** mechanism_info);
void writes_concentration(int type);
void point_mechanism(int type);
void artificial_cell(int type);
void net_receive(int type, int weight_count);
void prop_size(int type, int psize, int dpsize);
void dparam_semantic(int type, int index, const char* semantic);
void global_scalar(const char* name, double* value);
void global_scalar_field(int type, const char* field_name, const char* hoc_name, double* value);

}  // namespace mind_sim::micro::sim::nrn_registration_mirror
