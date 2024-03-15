/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file graph_executor.cc
 */
#include "graph_executor.h"

#include <tvm/runtime/container/map.h>
#include <tvm/runtime/container/string.h>
#include <tvm/runtime/data_type.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/profiling.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/serializer.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../file_utils.h"
#include "../texture.h"

namespace tvm {
namespace runtime {
namespace details {
inline size_t GetDataAlignment(const DLTensor& arr) {
  size_t align = (arr.dtype.bits / 8) * arr.dtype.lanes;
  if (align < kAllocAlignment) return kAllocAlignment;
  return align;
}
constexpr auto Is2DStorage = IsTextureStorage;
}  // namespace details

/*!
 * \brief Run all the operations one by one.
 */
void GraphExecutor::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) {op_execs_[i]();}
  }
}

void GraphExecutor::LoadRun(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadRun(&strm);
}

void GraphExecutor::LoadRun(dmlc::Stream* strm) {
  // setup the array and requirements.
  Map<String, NDArray> params = tvm::runtime::LoadParams(strm);
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    std::vector<size_t> indexs;
    std::vector<std::string> names;

    const auto& inode = nodes_[i];
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      for (auto& p : params) {
        size_t in_idx = GetInputIndex(p.first);
        if (in_idx < 0) continue;
        if (eid == this->entry_id(input_nodes_[in_idx], 0)) {
          indexs.push_back(in_idx);
          names.push_back(p.first);
          // data_entry_[eid].CopyFrom(p.second);
          // std::cout << "entry[" << eid << "]: " << static_cast<void*>(data_entry_[eid]->data) << " / " << std::addressof(data_entry_[eid]) <<std::endl;
        }
      }
    }

    IndexedSetupStorage(indexs);
    IndexedSetupOpExecs(indexs);

    std::cout << inode.name << ": ";
    for (size_t k : indexs) {
      std::cout << k << " / ";
    }
    for (std::string k : names) {
      std::cout << k << " / ";
    }
    std::cout << std::endl;
    // op_execs_[i]();
  }
}

/*!
 * \brief Initialize the graph executor with graph and device.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions for the host
 * processor.
 * \param devs The devices of the host and devices where graph nodes will be
 * executed on.
 * \param lookup_linked_param_func Linked parameter lookup function. Default is nullptr.
 */
void GraphExecutor::Init(const std::string& graph_json, tvm::runtime::Module module,
                         const std::vector<Device>& devs,
                         const PackedFunc lookup_linked_param_func) {
  std::cout << "Init" << std::endl;
  std::istringstream is(graph_json);
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  module_ = module;
  devices_ = devs;
  // lookup_linked_param_ 아무것도 없는 팩함수임 → 아마 linked param이 없는 걸로 생각하는 게 좋을 수도
  lookup_linked_param_ = lookup_linked_param_func;
  std::cout << std::addressof(lookup_linked_param_) << std::endl;
  if (lookup_linked_param_ == nullptr) {
    lookup_linked_param_ = PackedFunc(
        [this](TVMArgs args, TVMRetValue* rv) { this->DefaultLookupLinkedParam(args, rv); });
  }
  // this->SetupStorage();
  // this->SetupOpExecs();
  std::vector<size_t> indexs = {0};
  IndexedSetupStorage(indexs);
  IndexedSetupOpExecs(indexs);
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    const uint32_t nid = input_nodes_[i];
    std::string& name = nodes_[nid].name;
    input_map_[name] = i;
  }
  for (size_t i = 0; i < outputs_.size(); i++) {
    const uint32_t nid = outputs_[i].node_id;
    std::string& name = nodes_[nid].name;
    std::stringstream ss;
    ss << name << ":" << i;
    output_map_[ss.str()] = i;
  }
}

/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int GraphExecutor::GetInputIndex(const std::string& name) {
  auto it = input_map_.find(name);
  if (it != input_map_.end()) {
    return it->second;
  }
  return -1;
}

/*!
 * \brief Get the input info of Graph by parsing the input nodes.
 * \return The shape and dtype tuple.
 */
std::tuple<GraphExecutor::ShapeInfo, GraphExecutor::DtypeInfo> GraphExecutor::GetInputInfo() const {
  GraphExecutor::ShapeInfo shape_dict;
  GraphExecutor::DtypeInfo dtype_dict;
  for (uint32_t nid : input_nodes_) {
    CHECK_LE(nid, nodes_.size());
    std::string name = nodes_[nid].name;
    if (param_names_.find(name) == param_names_.end()) {
      CHECK_LE(nid, attrs_.shape.size());
      auto shape = attrs_.shape[nid];
      shape_dict.Set(name, ShapeTuple(shape));
      CHECK_LE(nid, attrs_.dltype.size());
      auto dtype = attrs_.dltype[nid];
      dtype_dict.Set(name, String(dtype));
    }
  }
  return std::make_tuple(shape_dict, dtype_dict);
}

/*!
 * \brief Get the output info of Graph by parsing the output nodes.
 * \return The shape and dtype tuple.
 */
std::tuple<GraphExecutor::ShapeInfo, GraphExecutor::DtypeInfo> GraphExecutor::GetOutputInfo()
    const {
  GraphExecutor::ShapeInfo shape_dict;
  GraphExecutor::DtypeInfo dtype_dict;
  for (auto out : outputs_) {
    uint32_t nid = out.node_id;
    CHECK_LE(nid, nodes_.size());
    std::string name = nodes_[nid].name;
    CHECK_LE(nid, attrs_.shape.size());
    auto shape = attrs_.shape[nid];
    shape_dict.Set(name, ShapeTuple(shape));
    CHECK_LE(nid, attrs_.dltype.size());
    auto dtype = attrs_.dltype[nid];
    dtype_dict.Set(name, String(dtype));
  }
  return std::make_tuple(shape_dict, dtype_dict);
}

/*!
 * \brief Get the output index given the name of output.
 * \param name The name of the output.
 * \return The index of output.
 */
int GraphExecutor::GetOutputIndex(const std::string& name) {
  auto it = output_map_.find(name);
  if (it != output_map_.end()) {
    return it->second;
  }
  return -1;
}
/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void GraphExecutor::SetInput(int index, DLTensor* data_in) {
  std::cout << "SetInput" << std::endl;

  ICHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  data_entry_[eid].CopyFrom(data_in);
}
/*!
 * \brief Check the legality of external DLTensor*.
 * \param external The external DLTensor*.
 * \param eid The data_enrty_ index.
 */
void GraphExecutor::CheckExternalDLTensor(const DLTensor* external, uint32_t eid) const {
  const DLTensor* internal = data_entry_[eid].operator->();

  ICHECK_EQ(data_alignment_[eid], details::GetDataAlignment(*external));
  ICHECK_EQ(reinterpret_cast<size_t>(static_cast<char*>(external->data) + external->byte_offset) %
                kAllocAlignment,
            0);
  ICHECK_EQ(internal->ndim, static_cast<size_t>(external->ndim));
  ICHECK_EQ(internal->device.device_type, external->device.device_type);
  ICHECK_EQ(internal->device.device_id, external->device.device_id);
  for (auto i = 0; i < external->ndim; ++i) {
    ICHECK_EQ(internal->shape[i], external->shape[i]);
  }
}
/*!
 * \brief set index-th input to the graph without copying the data.
 * \param index The input index.
 * \param data_ref The input data that is referred.
 */
void GraphExecutor::SetInputZeroCopy(int index, DLTensor* data_ref) {
  ICHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  // check the consistency of input
  CheckExternalDLTensor(data_ref, eid);
  // Update the data pointer for each argument of each op
  for (DLTensor* t : input_dltensors_[eid]) {
    t->data = static_cast<char*>(data_ref->data) + data_ref->byte_offset;
  }
}
/*!
 * \brief set index-th output to the graph without copying the data.
 * \param index The output index.
 * \param data_ref The output data that is referred.
 */
void GraphExecutor::SetOutputZeroCopy(int index, DLTensor* data_ref) {
  ICHECK_LT(static_cast<size_t>(index), outputs_.size());
  ICHECK_LT(static_cast<size_t>(index), output_dltensors_.size());
  const NodeEntry& output_node = outputs_[index];
  uint32_t output_node_eid = this->entry_id(output_node);

  // check the consistency of output
  CheckExternalDLTensor(data_ref, output_node_eid);

  // Update the data pointer for output op
  for (DLTensor* t : output_dltensors_[output_node_eid]) {
    t->data = static_cast<char*>(data_ref->data) + data_ref->byte_offset;
  }

  // Update the input of the op connected to the output
  for (DLTensor* t : both_output_opinput_dltensors_[output_node_eid]) {
    t->data = static_cast<char*>(data_ref->data) + data_ref->byte_offset;
  }
}
/*!
 * \brief Get the number of outputs
 *
 * \return The number of outputs from graph.
 */
int GraphExecutor::NumOutputs() const { return outputs_.size(); }
/*!
 * \brief Get the number of inputs
 *
 * \return The number of inputs to the graph.
 */
int GraphExecutor::NumInputs() const { return input_nodes_.size(); }
/*!
 * \brief Return NDArray for given input index.
 * \param index The input index.
 *
 * \return NDArray corresponding to given input node index.
 */
NDArray GraphExecutor::GetInput(int index) const {
  std::cout << "getinput 1" << std::endl;
  ICHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  std::cout << "getinput 2" << std::endl;
  uint32_t eid = this->entry_id(input_nodes_[index], 0);
  std::cout << "getinput 3" << std::endl;
  return data_entry_[eid];
}
/*!
 * \brief Return NDArray for given output index.
 * \param index The output index.
 *
 * \return NDArray corresponding to given output node index.
 */
NDArray GraphExecutor::GetOutput(int index) const {
  ICHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  return data_entry_[eid];
}
/*!
 * \brief Copy index-th output to data_out.
 * \param index The output index.
 * \param data_out the output data.
 */
void GraphExecutor::CopyOutputTo(int index, DLTensor* data_out) {
  ICHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);

  // Check the shapes to avoid receiving in different dimension but same size.
  const NDArray& data = data_entry_[eid];
  ICHECK_EQ(data->ndim, data_out->ndim);
  for (int32_t j = 0; j < data->ndim; ++j) {
    ICHECK_EQ(data->shape[j], data_out->shape[j]);
  }

  data_entry_[eid].CopyTo(data_out);
}

/*!
 * \brief Load parameters from parameter blob.
 * \param param_blob A binary blob of parameter.
 */
void GraphExecutor::LoadParams(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

// parameter를 op에 연결하는 것 연산 안에 runtime::LoadParams는 stream에서 Parameter dictionary 만드는 함수
void GraphExecutor::LoadParams(dmlc::Stream* strm) {
  std::cout << "LoadParams" << std::endl;
  Map<String, NDArray> params = ::tvm::runtime::LoadParams(strm);
  // param_names : parameter들 이름 리스트
  for (auto& p : params) {
    // p.first : parameter 이름, GetInputIndex로 이름으로 input_nodes_의 인덱스 따옴
    param_names_.insert(p.first);
    int in_idx = GetInputIndex(p.first);
    if (in_idx < 0) continue;
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    data_entry_[eid].CopyFrom(p.second);
  }
}

void GraphExecutor::ShareParams(const GraphExecutor& other, dmlc::Stream* strm) {
  std::cout << "ShareParams" << std::endl;
  uint64_t header, reserved;
  ICHECK(strm->Read(&header)) << "Invalid parameters file format";
  ICHECK(header == kTVMNDArrayListMagic) << "Invalid parameters file format";
  ICHECK(strm->Read(&reserved)) << "Invalid parameters file format";
  std::vector<std::string> names;
  ICHECK(strm->Read(&names)) << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  ICHECK(size == names.size()) << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    if (in_idx < 0) continue;
    uint32_t eid = this->entry_id(input_nodes_[in_idx], 0);
    ICHECK_LT(eid, data_entry_.size());
    ICHECK_EQ(data_entry_[eid].use_count(), 1);
    data_entry_[eid] = other.GetInput(GetInputIndex(names[i]));
    ICHECK_GT(data_entry_[eid].use_count(), 1);
    const DLTensor* tmp = data_entry_[eid].operator->();
    data_alignment_[eid] = details::GetDataAlignment(*tmp);
  }
  this->SetupOpExecs();
}

void GraphExecutor::LinkedNDArrayDeleter(Object* container) {
  // container is the NDArray::Container which needs to get deleted.
  // The data member points to global const memory, so it does not need deleting.
  delete static_cast<NDArray::Container*>(container);
}

void GraphExecutor::DefaultLookupLinkedParam(TVMArgs args, TVMRetValue* rv) {
  Module mod = args[0];
  int64_t storage_id = args[1];
  DLTensor* template_tensor = args[2];
  Device dev = args[3];
  // Get pre-linked parameter lookup function, if it was generated. When pf == nullptr, no linked
  // params are present.
  if (!module_lookup_linked_param_valid_) {
    module_lookup_linked_param_ =
        mod.GetFunction(::tvm::runtime::symbol::tvm_lookup_linked_param, true);
  }
  if (module_lookup_linked_param_ == nullptr) {
    *rv = nullptr;
    return;
  }

  TVMRetValue opaque_handle = module_lookup_linked_param_(storage_id);
  if (opaque_handle.type_code() == kTVMNullptr) {
    *rv = nullptr;
    return;
  }

  std::vector<int64_t> shape_vec{template_tensor->shape,
                                 template_tensor->shape + template_tensor->ndim};

  auto* container = new NDArray::Container(static_cast<void*>(opaque_handle), shape_vec,
                                           template_tensor->dtype, dev);
  container->SetDeleter(GraphExecutor::LinkedNDArrayDeleter);
  *rv = NDArray(GetObjectPtr<Object>(container));
}

void GraphExecutor::IndexedSetupStorage(std::vector<size_t> indexs) {
  if (indexs.empty()) { return; }
  // Grab saved optimization plan from graph.
  std::cout << "SetupStorage" << std::endl;

  std::vector<DLDataType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    // 전부 float32
    // std::cout << s_type << std::endl;
    vtype.push_back(tvm::runtime::String2DLDataType(s_type));
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i : indexs) {
    std::cout << i << std::endl;
    int storage_id = attrs_.storage_id[i];
    std::string storage_scope = attrs_.storage_scope.empty() ? "" : attrs_.storage_scope[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(devices_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    // std::cout << "i: " << i << " / storage_id: " << storage_id << std::endl;
    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {-1, {0}, {}});
    } else {
      ICHECK(pool_entry[sid].device_type == -1 || pool_entry[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    TVMRetValue lookup_rv;
    {
      // shape_vec - 1/3/224/224 같이 데이터 모양들 
      std::vector<int64_t> shape_vec{attrs_.shape[i].begin(), attrs_.shape[i].end()};
      // for (const auto& as : shape_vec) {
      //   std::cout << as << " / ";
      // }
      // std::cout << std::endl;
      // 정해진 크기의 DLTensor 만드는 것 같은데
      DLTensor template_tensor{nullptr,  Device{kDLCPU, 0}, static_cast<int>(shape_vec.size()),
                               vtype[i], shape_vec.data(),  nullptr,
                               0};
      // 여기선 데이터 메모리 주소가 할당되지는 않는 듯?
      // std::cout << "index[" << i << "]: " << static_cast<void*>(template_tensor.data) << std::endl;
      lookup_rv = lookup_linked_param_(module_, sid, &template_tensor, devices_[0]);
    }
    if (lookup_rv.type_code() != kTVMNullptr) {
      pool_entry[sid].linked_param = lookup_rv;
    }
    pool_entry[sid].param_data_entry = i;
    pool_entry[sid].device_type = device_type;
    pool_entry[sid].scope = storage_scope;

    // vtype이 data 타입임 → 전부 float32
    DLDataType t = vtype[i];
    if (!details::Is2DStorage(storage_scope)) {
      size_t size = 1;
      for (int64_t sz : attrs_.shape[i]) {
        size *= static_cast<size_t>(sz);
      }
      // std::cout << t.lanes << std::endl;
      // t.lanes이 vector 같이 여러개 있는 애들의 개수 여기서는 전부 1
      // t.bits는 vtype에 저장되어 있는 데이터 타입의 비트 수 - 여기서는 float32이니까 32
      size_t bits = t.bits * t.lanes;
      ICHECK(bits % 8U == 0U || bits == 1U || bits == 4U);
      int64_t bytes = ((bits + 7U) / 8U) * size;
      // std::cout << "sid: " << sid << " / " << pool_entry[sid].shape[0] << " / " << std::max(pool_entry[sid].shape[0], bytes) << std::endl;
      // 100~105번 sid는 op 진행할 때 재활용이 되는 것 같긴 한데, 사이즈가 다른 문제는 그냥 큰 거를 사용하는 듯.
      pool_entry[sid].shape[0] = std::max(pool_entry[sid].shape[0], bytes);
      pool_entry[sid].dtype = DLDataType{kDLFloat, 32, 1};
    } else {
      if (pool_entry[sid].shape.size() == 1) {
        pool_entry[sid].shape.resize(3, 0);
      }
      size_t axis = runtime::DefaultTextureLayoutSeparator(attrs_.shape[i].size(), storage_scope);
      auto shape = ApplyTexture2DFlattening<int64_t>(attrs_.shape[i], attrs_.shape[i].size(), axis);
      pool_entry[sid].shape[0] = std::max(pool_entry[sid].shape[0], shape.height);
      pool_entry[sid].shape[1] = std::max(pool_entry[sid].shape[1], shape.width);
      CHECK(pool_entry[sid].shape[2] == 0 || pool_entry[sid].shape[2] == shape.channel)
          << pool_entry[sid].shape[2] << " != " << shape.channel
          << ",  texture channel length must be consistent within a storage pool";
      pool_entry[sid].shape[2] = shape.channel;
      CHECK(pool_entry[sid].dtype.bits == 0 || TypeEqual(pool_entry[sid].dtype, t))
          << DLDataType2String(pool_entry[sid].dtype) << " != " << DLDataType2String(t)
          << ", pool entry for 2d texure allocations must be of the same type;"
          << " downstream error from memory planner likely";
      pool_entry[sid].dtype = t;
    }
  }

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    // This for loop is very fast since there are usually only a couple of
    // devices available on the same hardware.
    const auto& cit = std::find_if(devices_.begin(), devices_.end(), [&pit](const Device& d) {
      return pit.device_type == static_cast<int>(d.device_type);
    });
    Device dev = cit == devices_.end() ? devices_[0] : *cit;
    if (pit.linked_param.defined()) {
      // 이쪽 실행 안 됨 전부 else
      storage_pool_.push_back(pit.linked_param);
    } else {
      std::vector<int64_t> shape = pit.shape;
      if (shape.size() == 1) {
        shape[0] = (shape[0] + 3) / 4;
      }
      Optional<String> mem_scope;
      if (!pit.scope.empty()) {
        mem_scope = String(pit.scope);
      }
      storage_pool_.push_back(MemoryManager::GetOrCreateAllocator(dev, AllocatorType::kNaive)
                                  ->Empty(shape, pit.dtype, dev, mem_scope));
    }
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  data_alignment_.resize(num_node_entries());
  // sid_to_eid has a size of storage_id's size, which is the size of storage_pool_.
  sid_to_eid_.resize(storage_pool_.size());
  for (size_t i : indexs) {
    int storage_id = attrs_.storage_id[i];
    // Update "storage_id -> entry_id" pair.
    sid_to_eid_[storage_id].push_back(i);

    ICHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    // storage_pool_[storage_id] -> NDArray, CreateView -> Create a NDArray that shares the data memory with the current one
    // 여기까지는 data_entry_에 할당 안 됨, storage_pool_이랑 data의 주소가 같다
    data_entry_[i] = storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
    // std::cout << "entry[" << i << "]: " << static_cast<void*>(data_entry_[i]->data) << " / " << static_cast<void*>(storage_pool_[storage_id]->data) << std::endl;

    const DLTensor* tmp = data_entry_[i].operator->();
    data_alignment_[i] = details::GetDataAlignment(*tmp);
  }
}

void GraphExecutor::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::cout << "SetupStorage" << std::endl;

  std::vector<DLDataType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    // 전부 float32
    // std::cout << s_type << std::endl;
    vtype.push_back(tvm::runtime::String2DLDataType(s_type));
  }

  // Size and device type of each storage pool entry.
  std::vector<PoolEntry> pool_entry;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    std::string storage_scope = attrs_.storage_scope.empty() ? "" : attrs_.storage_scope[i];
    // Use the fallback device if no device index is available.
    int device_type = static_cast<int>(devices_[0].device_type);
    if (!attrs_.device_index.empty()) {
      device_type = attrs_.device_index[i];
    }
    // std::cout << "i: " << i << " / storage_id: " << storage_id << std::endl;
    uint32_t sid = static_cast<uint32_t>(storage_id);
    if (sid >= pool_entry.size()) {
      pool_entry.resize(sid + 1, {-1, {0}, {}});
    } else {
      ICHECK(pool_entry[sid].device_type == -1 || pool_entry[sid].device_type == device_type)
          << "The same pool entry cannot be assigned to multiple devices";
    }
    TVMRetValue lookup_rv;
    {
      // shape_vec - 1/3/224/224 같이 데이터 모양들 
      std::vector<int64_t> shape_vec{attrs_.shape[i].begin(), attrs_.shape[i].end()};
      // for (const auto& as : shape_vec) {
      //   std::cout << as << " / ";
      // }
      // std::cout << std::endl;
      // 정해진 크기의 DLTensor 만드는 것 같은데
      DLTensor template_tensor{nullptr,  Device{kDLCPU, 0}, static_cast<int>(shape_vec.size()),
                               vtype[i], shape_vec.data(),  nullptr,
                               0};
      // 여기선 데이터 메모리 주소가 할당되지는 않는 듯?
      // std::cout << "index[" << i << "]: " << static_cast<void*>(template_tensor.data) << std::endl;
      lookup_rv = lookup_linked_param_(module_, sid, &template_tensor, devices_[0]);
    }
    if (lookup_rv.type_code() != kTVMNullptr) {
      pool_entry[sid].linked_param = lookup_rv;
    }
    pool_entry[sid].param_data_entry = i;
    pool_entry[sid].device_type = device_type;
    pool_entry[sid].scope = storage_scope;

    // vtype이 data 타입임 → 전부 float32
    DLDataType t = vtype[i];
    if (!details::Is2DStorage(storage_scope)) {
      size_t size = 1;
      for (int64_t sz : attrs_.shape[i]) {
        size *= static_cast<size_t>(sz);
      }
      // std::cout << t.lanes << std::endl;
      // t.lanes이 vector 같이 여러개 있는 애들의 개수 여기서는 전부 1
      // t.bits는 vtype에 저장되어 있는 데이터 타입의 비트 수 - 여기서는 float32이니까 32
      size_t bits = t.bits * t.lanes;
      ICHECK(bits % 8U == 0U || bits == 1U || bits == 4U);
      int64_t bytes = ((bits + 7U) / 8U) * size;
      // std::cout << "sid: " << sid << " / " << pool_entry[sid].shape[0] << " / " << std::max(pool_entry[sid].shape[0], bytes) << std::endl;
      // 100~105번 sid는 op 진행할 때 재활용이 되는 것 같긴 한데, 사이즈가 다른 문제는 그냥 큰 거를 사용하는 듯.
      pool_entry[sid].shape[0] = std::max(pool_entry[sid].shape[0], bytes);
      pool_entry[sid].dtype = DLDataType{kDLFloat, 32, 1};
    } else {
      if (pool_entry[sid].shape.size() == 1) {
        pool_entry[sid].shape.resize(3, 0);
      }
      size_t axis = runtime::DefaultTextureLayoutSeparator(attrs_.shape[i].size(), storage_scope);
      auto shape = ApplyTexture2DFlattening<int64_t>(attrs_.shape[i], attrs_.shape[i].size(), axis);
      pool_entry[sid].shape[0] = std::max(pool_entry[sid].shape[0], shape.height);
      pool_entry[sid].shape[1] = std::max(pool_entry[sid].shape[1], shape.width);
      CHECK(pool_entry[sid].shape[2] == 0 || pool_entry[sid].shape[2] == shape.channel)
          << pool_entry[sid].shape[2] << " != " << shape.channel
          << ",  texture channel length must be consistent within a storage pool";
      pool_entry[sid].shape[2] = shape.channel;
      CHECK(pool_entry[sid].dtype.bits == 0 || TypeEqual(pool_entry[sid].dtype, t))
          << DLDataType2String(pool_entry[sid].dtype) << " != " << DLDataType2String(t)
          << ", pool entry for 2d texure allocations must be of the same type;"
          << " downstream error from memory planner likely";
      pool_entry[sid].dtype = t;
    }
  }

  // Allocate the space.
  for (const auto& pit : pool_entry) {
    // This for loop is very fast since there are usually only a couple of
    // devices available on the same hardware.
    const auto& cit = std::find_if(devices_.begin(), devices_.end(), [&pit](const Device& d) {
      return pit.device_type == static_cast<int>(d.device_type);
    });
    Device dev = cit == devices_.end() ? devices_[0] : *cit;
    if (pit.linked_param.defined()) {
      // 이쪽 실행 안 됨 전부 else
      storage_pool_.push_back(pit.linked_param);
    } else {
      std::vector<int64_t> shape = pit.shape;
      if (shape.size() == 1) {
        shape[0] = (shape[0] + 3) / 4;
      }
      Optional<String> mem_scope;
      if (!pit.scope.empty()) {
        mem_scope = String(pit.scope);
      }
      storage_pool_.push_back(MemoryManager::GetOrCreateAllocator(dev, AllocatorType::kNaive)
                                  ->Empty(shape, pit.dtype, dev, mem_scope));
    }
  }

  // Assign the pooled entries. A unified memory pool is used to simplifiy
  // memory assignment for each node entry. The allocated memory on each device
  // is mapped to this pool.
  data_entry_.resize(num_node_entries());
  data_alignment_.resize(num_node_entries());
  // sid_to_eid has a size of storage_id's size, which is the size of storage_pool_.
  sid_to_eid_.resize(storage_pool_.size());
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    // Update "storage_id -> entry_id" pair.
    sid_to_eid_[storage_id].push_back(i);

    ICHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    // storage_pool_[storage_id] -> NDArray, CreateView -> Create a NDArray that shares the data memory with the current one
    // 여기까지는 data_entry_에 할당 안 됨, storage_pool_이랑 data의 주소가 같다
    data_entry_[i] = storage_pool_[storage_id].CreateView(attrs_.shape[i], vtype[i]);
    // std::cout << "entry[" << i << "]: " << static_cast<void*>(data_entry_[i]->data) << " / " << static_cast<void*>(storage_pool_[storage_id]->data) << std::endl;

    const DLTensor* tmp = data_entry_[i].operator->();
    data_alignment_[i] = details::GetDataAlignment(*tmp);
  }
}

void GraphExecutor::SetupPageTable() {
  uint32_t max_input = 0;
  for (size_t i = 0; i < nodes_.size(); i++) {
    const auto& inode = nodes_[i];

    if (inode.op_type == "null") {
      // uint32_t eid = this->entry_id(i, 0);
      // std::cout << "entry[" << eid << "]: " << static_cast<void*>(data_entry_[eid]->data) << " / " << std::addressof(data_entry_[eid]) <<std::endl;
      continue;
    }
    if (inode.param.num_inputs > max_input) { max_input = inode.param.num_inputs; }
  }

  std::cout << "max_input: " << max_input << std::endl;

  // for (size_t i = 0; i < input_nodes_.size(); i++) {
  //   const uint32_t nid = input_nodes_[i];

  //   std::cout << "i: " << i << " / nid: " << nid << std::endl;
  // }
}

void GraphExecutor::IndexedSetupOpExecs(std::vector<size_t> indexs) {
  if (indexs.empty()) { return; }

  std::cout << "SetupOpExecs" << std::endl;

  op_execs_.resize(this->GetNumOfNodes());
  input_dltensors_.resize(num_node_entries());
  output_dltensors_.resize(num_node_entries());
  both_output_opinput_dltensors_.resize(num_node_entries());

  // input node entry index 만드는 과정
  std::unordered_set<uint32_t> input_node_eids;
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    uint32_t nid = input_nodes_[i];
    input_node_eids.insert(entry_id(nid, 0));
    // 다 같음
  }

  // output node entry index 만드는 과정
  std::unordered_set<uint32_t> output_node_eids;
  for (size_t i = 0; i < outputs_.size(); i++) {
    output_node_eids.insert(entry_id(outputs_[i]));
    // 1개
  }

  // setup the array and requirements.
  for (size_t nid : indexs) {
    // 연산 노드의 node
    nid = (uint32_t)nid;

    const auto& inode = nodes_[nid];

    if (inode.op_type == "null") {
      continue;
    }
    std::vector<DLTensor*> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      // op_type은 tvm_op, e.index는 전부 0, eid = e.node_id + e.index = e.node_id, 0 ~ 143 까지 나옴
      // push_back은 vector 끝에 요소 추가하는 함수
      args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
    }
    ICHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    std::shared_ptr<OpArgs> op_args = nullptr;
    // args는 input, output data_entry_ 위치
    std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args);

    // dltensors 얘네 없어도 잘 돌아가는데 뭐지? 왜 있는 거지..
    for (size_t i = 0; i < inode.inputs.size(); i++) {
      uint32_t input_eid = this->entry_id(inode.inputs[i]);
      // check if op input is model input
      if (input_node_eids.count(input_eid) > 0) {
        input_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));

        // Data entry who has the same storage_id should also be pushed into "input_dltensors" and
        // being able to be updated by "SetInputZeroCopy()". This is to handle the situation that a
        // "relay.reshape" follows immediately after input and input dltensor and reshape's output
        // dltensor point to the same data_entry.
        auto storage_id = attrs_.storage_id[input_eid];
        for (auto eid : sid_to_eid_[storage_id]) {
          input_dltensors_[input_eid].push_back(
              const_cast<DLTensor*>(data_entry_[eid].operator->()));
        }
      }
      // check if any model output is the input of the op
      if (output_node_eids.count(input_eid) > 0) {
        both_output_opinput_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }

    for (uint32_t i = inode.inputs.size(); i < inode.inputs.size() + inode.param.num_outputs; ++i) {
      uint32_t output_eid = this->entry_id(nid, i - inode.inputs.size());
      // check if op output is model output
      if (output_node_eids.count(output_eid) > 0) {
        output_dltensors_[output_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }
  }
}

void GraphExecutor::SetupOpExecs() {
  std::cout << "SetupOpExecs" << std::endl;

  op_execs_.resize(this->GetNumOfNodes());
  input_dltensors_.resize(num_node_entries());
  output_dltensors_.resize(num_node_entries());
  both_output_opinput_dltensors_.resize(num_node_entries());

  // input node entry index 만드는 과정
  std::unordered_set<uint32_t> input_node_eids;
  for (size_t i = 0; i < input_nodes_.size(); i++) {
    uint32_t nid = input_nodes_[i];
    input_node_eids.insert(entry_id(nid, 0));
    // 다 같음
  }

  // output node entry index 만드는 과정
  std::unordered_set<uint32_t> output_node_eids;
  for (size_t i = 0; i < outputs_.size(); i++) {
    output_node_eids.insert(entry_id(outputs_[i]));
    // 1개
  }

  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->GetNumOfNodes(); ++nid) {
    // 연산 노드의 node
    const auto& inode = nodes_[nid];

    if (inode.op_type == "null") {
      continue;
    }
    std::vector<DLTensor*> args;
    for (const auto& e : inode.inputs) {
      uint32_t eid = this->entry_id(e);
      // op_type은 tvm_op, e.index는 전부 0, eid = e.node_id + e.index = e.node_id, 0 ~ 143 까지 나옴
      // push_back은 vector 끝에 요소 추가하는 함수
      args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->entry_id(nid, index);
      args.push_back(const_cast<DLTensor*>(data_entry_[eid].operator->()));
    }
    ICHECK(inode.op_type == "tvm_op") << "Can only take tvm_op as op";

    std::shared_ptr<OpArgs> op_args = nullptr;
    // args는 input, output data_entry_ 위치
    std::tie(op_execs_[nid], op_args) = CreateTVMOp(inode.param, args);

    // dltensors 얘네 없어도 잘 돌아가는데 뭐지? 왜 있는 거지..
    for (size_t i = 0; i < inode.inputs.size(); i++) {
      uint32_t input_eid = this->entry_id(inode.inputs[i]);
      // check if op input is model input
      if (input_node_eids.count(input_eid) > 0) {
        input_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));

        // Data entry who has the same storage_id should also be pushed into "input_dltensors" and
        // being able to be updated by "SetInputZeroCopy()". This is to handle the situation that a
        // "relay.reshape" follows immediately after input and input dltensor and reshape's output
        // dltensor point to the same data_entry.
        auto storage_id = attrs_.storage_id[input_eid];
        for (auto eid : sid_to_eid_[storage_id]) {
          input_dltensors_[input_eid].push_back(
              const_cast<DLTensor*>(data_entry_[eid].operator->()));
        }
      }
      // check if any model output is the input of the op
      if (output_node_eids.count(input_eid) > 0) {
        both_output_opinput_dltensors_[input_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }

    for (uint32_t i = inode.inputs.size(); i < inode.inputs.size() + inode.param.num_outputs; ++i) {
      uint32_t output_eid = this->entry_id(nid, i - inode.inputs.size());
      // check if op output is model output
      if (output_node_eids.count(output_eid) > 0) {
        output_dltensors_[output_eid].push_back(
            static_cast<DLTensor*>(op_args->arg_values[i].v_handle));
      }
    }
  }
}

std::pair<std::function<void()>, std::shared_ptr<GraphExecutor::OpArgs>> GraphExecutor::CreateTVMOp(
    const TVMOpParam& param, const std::vector<DLTensor*>& args) {
  std::shared_ptr<GraphExecutor::OpArgs> arg_ptr = std::make_shared<GraphExecutor::OpArgs>();
  // setup address.
  arg_ptr->args = args;
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    // input, output DLTensor 포인터
    DLTensor* t = arg_ptr->args[i];
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    // kTVMDLTensorHandle → 7
    arg_ptr->arg_tcodes.push_back(kTVMDLTensorHandle);
    // 연산들은 flatten data가 없어서 밑에 실행이 안 되는구나
    if (param.flatten_data) {
      // 총 요소 개수 구함 (3,2,1) -> 6 곱해서
      arg_ptr->shape_data[i] =
          std::accumulate(t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      // 차원 1개의 1차원으로 만들어버리네?
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }

  if (param.func_name == "__nop") {
    return {[]() {}, arg_ptr};
  } else if (param.func_name == "__copy") {
    // Perform cross device data copy.
    // Directly copy data from the input to the output.
    // TODO(mbs): device_copy cleanup.
    auto fexec = [arg_ptr]() {
      DLTensor* from = static_cast<DLTensor*>(arg_ptr->arg_values[0].v_handle);
      DLTensor* to = static_cast<DLTensor*>(arg_ptr->arg_values[1].v_handle);
      TVM_CCALL(TVMArrayCopyFromTo(from, to, nullptr));
    };
    return {fexec, arg_ptr};
  }

  // Get compiled function from the module that contains both host and device
  // code.
  // module_이 operation 코드인데, 거기에 데이터를 넣어서 실행시키는 람다함수를 보내는 식이구나
  // 넣는 데이터가 어떤 식인지만 파악하자
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, true);
  ICHECK(pf != nullptr) << "no such function in module: " << param.func_name;

  auto fexec = [arg_ptr, pf]() {
    TVMRetValue rv;
    // .data()는 첫 포인터 가르킴, tcodes는 어떤 역할인지 가늠이 안 가네 → FFI에서 사용되는 인자 유형 코드, 팩함수 호출 할 때 보고 판단함
    TVMArgs targs(arg_ptr->arg_values.data(), arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return {fexec, arg_ptr};
}

PackedFunc GraphExecutor::GetFunction(const String& name, const ObjectPtr<Object>& sptr_to_self) {
  // Return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int in_idx = this->GetInputIndex(args[0].operator String());
        if (in_idx >= 0) this->SetInput(in_idx, args[1]);
      } else {
        this->SetInput(args[0], args[1]);
      }
    });
  } else if (name == "set_input_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int in_idx = this->GetInputIndex(args[0].operator String());
        if (in_idx >= 0) this->SetInputZeroCopy(in_idx, args[1]);
      } else {
        this->SetInputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "set_output_zero_copy") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (String::CanConvertFrom(args[0])) {
        int out_idx = this->GetOutputIndex(args[0].operator String());
        if (out_idx >= 0) this->SetOutputZeroCopy(out_idx, args[1]);
      } else {
        this->SetOutputZeroCopy(args[0], args[1]);
      }
    });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      if (args.num_args == 2) {
        this->CopyOutputTo(args[0], args[1]);
      } else {
        int out_idx = -1;
        if (String::CanConvertFrom(args[0])) {
          for (size_t i = 0; i < outputs_.size(); i++) {
            std::string& name = nodes_[outputs_[i].node_id].name;
            if (args[0].operator String() == name) {
              out_idx = i;
            }
          }
          CHECK(out_idx != -1) << "Invalid output node:" << args[0].operator String();
        } else {
          out_idx = args[0];
        }
        *rv = this->GetOutput(out_idx);
      }
    });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      int in_idx = 0;
      if (String::CanConvertFrom(args[0])) {
        in_idx = this->GetInputIndex(args[0].operator String());
      } else {
        in_idx = args[0];
      }
      if (in_idx >= 0) {
        *rv = this->GetInput(in_idx);
      }
    });
  } else if (name == "get_num_outputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumOutputs(); });
  } else if (name == "get_num_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { *rv = this->NumInputs(); });
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) { this->Run(); });
  } else if (name == "load_run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->LoadRun(args[0].operator std::string());
    });
  } else if (name == "run_from_inputs") {
    return PackedFunc(
        [sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
          CHECK(args.size() % 2 == 0)
              << "Number of arguments to run_from_inputs must be an even number of key-value pairs";
          Device host{static_cast<DLDeviceType>(args[0].operator int()), args[1].operator int()};
          for (int i = 2; i < args.size(); i += 2) {
            if (String::CanConvertFrom(args[i])) {
              int in_idx = this->GetInputIndex(args[i].operator String());
              if (in_idx >= 0) {
                this->SetInput(in_idx, args[i + 1]);
              } else {
                LOG(FATAL) << args[i].operator String() << " is not a valid input name";
              }
            } else {
              this->SetInput(args[i], args[i + 1]);
            }
          }
          this->Run();
          Array<NDArray> outputs;
          for (int i = 0; i < this->NumOutputs(); i++) {
            NDArray out = this->GetOutput(i);
            NDArray a = NDArray::Empty(out.Shape(), out.DataType(), host);
            a.CopyFrom(out);
            outputs.push_back(a);
          }
          *rv = outputs;
        });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      this->LoadParams(args[0].operator std::string());
    });
  } else if (name == "share_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      const auto& module = args[0].operator Module();
      ICHECK_EQ(module.operator->()->type_key(), std::string("GraphExecutor"));
      const auto& param_blob = args[1].operator std::string();
      dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
      this->ShareParams(dynamic_cast<const GraphExecutor&>(*module.operator->()), &strm);
    });
  } else if (name == "get_input_index") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      CHECK(String::CanConvertFrom(args[0])) << "Input key is not a string";
      *rv = this->GetInputIndex(args[0].operator String());
    });
  } else if (name == "get_input_info") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      auto [shape_info, dtype_info] = this->GetInputInfo();
      Map<String, ObjectRef> input_info;
      input_info.Set("shape", shape_info);
      input_info.Set("dtype", dtype_info);
      *rv = input_info;
    });
  } else if (name == "get_output_info") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
      auto [shape_info, dtype_info] = this->GetOutputInfo();
      Map<String, ObjectRef> input_info;
      input_info.Set("shape", shape_info);
      input_info.Set("dtype", dtype_info);
      *rv = input_info;
    });
  } else {
    return PackedFunc();
  }
}

Module GraphExecutorCreate(const std::string& sym_json, const tvm::runtime::Module& m,
                           const std::vector<Device>& devs,
                           const PackedFunc lookup_linked_param_func) {
  auto exec = make_object<GraphExecutor>();
  exec->Init(sym_json, m, devs, lookup_linked_param_func);
  return Module(exec);
}

// Get all devices for the host and other runtime devices.
std::vector<Device> GetAllDevice(const TVMArgs& args, int dev_start_arg) {
  // Reserve the first item as the fallback device.
  std::vector<Device> ret;
  Device dev;
  for (int i = dev_start_arg; i < args.num_args; i += 2) {
    int dev_type = args[i];
    dev.device_type = static_cast<DLDeviceType>(dev_type);
    dev.device_id = args[i + 1];
    ret.push_back(dev);
  }
  return ret;
}

// 4-argument version is currently reserved to keep support of calling
// from tvm4j and javascript, since they don't have heterogeneous
// execution support yet. For heterogenenous execution, at least 5 arguments will
// be passed in. The third one is the number of devices.
// Eventually, we will only probably pass Device for all the languages.
TVM_REGISTER_GLOBAL("tvm.graph_executor.create").set_body([](TVMArgs args, TVMRetValue* rv) {
  ICHECK_GE(args.num_args, 4) << "The expected number of arguments for graph_executor.create is "
                                 "at least 4, but it has "
                              << args.num_args;
  PackedFunc lookup_linked_param_func;
  int dev_start_arg = 2;
  if (args[2].type_code() == kTVMPackedFuncHandle) {
    lookup_linked_param_func = args[2];
    dev_start_arg++;
  }
  const auto& devices = GetAllDevice(args, dev_start_arg);
  *rv = GraphExecutorCreate(args[0], args[1], devices, lookup_linked_param_func);
});
}  // namespace runtime
}  // namespace tvm
