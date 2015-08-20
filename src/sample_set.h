#pragma once
#include <memory>
#include <dmlc/data.h>

namespace admm {

  /*!
   * \brief A wrapper of the sample data set.
   *
   *  NOTE: Not supposed to be multi-threaded.
   */
class SampleSet {
  typedef uint32_t IndexType;
  typedef ::dmlc::RowBlockIter<IndexType> RBIter;
 public:
  /*!
   * \brief Initialize the instance of the sample set by specifying the 
   * data file with libsvm format on hdfs.
   *
   * It does LOAD the data here.
   *
   * \param uri the uri (hdfs://nn/path/to/it) of the files
   * \param part_index split/part id of the current input
   */
  bool Initialize(const std::string& uri,
      uint32_t part_index,
      uint32_t num_parts);

  /*!
   * \brief Iterate to next item. Return false if there's no more item.
   */
  bool Next();

  /*!
   * \brief Get the value of the current iterated item.
   */
  ::dmlc::Row<IndexType> GetData();

  /*!
   * \brief Reset the dataset iter
   */
  void Rewind();

  SampleSet();
  virtual ~SampleSet();
 protected:
  std::unique_ptr<RBIter> rbiter_;

  std::size_t rb_size_;
  std::size_t current_index_;
};

}  // namespace admm
