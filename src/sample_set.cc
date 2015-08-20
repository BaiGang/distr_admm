
#include "sample_set.h"
#include <dmlc/logging.h>

namespace admm {

SampleSet::SampleSet() {
}

SampleSet::~SampleSet() {
}

bool SampleSet::Initialize(const std::string& uri,
    uint32_t part_index, uint32_t num_parts) {
  auto rbiter = ::dmlc::RowBlockIter<IndexType>::Create(
      uri.c_str(), part_index, num_parts, "libsvm");
  if (rbiter == NULL) {
    LOG(ERROR) << "Failed open " << uri << " as libsvm formatted file.";
    return false;
  }
  rbiter_.reset(rbiter);
  return true;
}

bool SampleSet::Next() {
  ++current_index_;
  if (current_index_ >= rb_size_) {
    if (!rbiter_->Next()) {
      return false;
    }
    rb_size_ = rbiter_->Value().size;
    current_index_ = 0;
  }
  return true;
}

::dmlc::Row<SampleSet::IndexType> SampleSet::GetData() {
  return rbiter_->Value()[current_index_];
}

void SampleSet::Rewind() {
  rbiter_->BeforeFirst();
  current_index_ = -1;
  rb_size_ = -1;
}

}  // namespace admm
