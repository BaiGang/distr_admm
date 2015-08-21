#pragma once
#include <cmath>
#include <vector>
#include <dmlc/data.h>
#include "sample_set.h"
#include "config.h"

namespace ftrl{
	
/*!
 * \brief FTRL_Proximal solver.
 * 
 *  For training Logistic regression with L_1 and 
 *  L_2 regularization
 */

class FTRL_Solver {
 public:
  typedef float real_t;
  typedef std::size_t IndexType;
  typedef ::dmlc::Row<IndexType> Row;
  typedef ::admm::SampleSet SampleSet;
  /*!
   * \brief Initiate with l_1, l_2, num_iter, alpha, beta
   */
  FTRL_Solver(::admm::FtrlConfig & params);
  FTRL_Solver(real_t lambda_1,
			  real_t lambda_2,
			  real_t alpha_init,
			  real_t  beta_init,
			  std::size_t niter_init,
			  std::size_t dim_init);
  ~FTRL_Solver();
  
  /*!
   * \brief Compute the value returned by logistic function. 
   *
   * \param x the processed training instance.
   * \param offset the intercept weights.
   * \return prediction of instance x.
   */
  real_t Predict(Row& x);
  real_t Predict(Row& x, std::vector<real_t>& offset);

  /*!
   * \brief assignment for the weights
   *
   * \param x the target which the weights are assigned with.
   */
  void Assign(const std::vector<real_t>& x);
  /*!
   * \brief update model using ftrl algorithm
   *
   * \param x the processed training instance
   * \param predict the prediction of x
   */
  void Update(const Row& x, real_t predict);

  /*!
   * \brief get the weight solution
   */
  const std::vector<real_t>& GetWeight();

  /*!
   * \brief process the update with specified passes
   *
   * \param the samples set corresponding the model
   */
  void Run(SampleSet& sample_set,std::vector<real_t>& offset);

 protected:

  /*!
   * \brief the weights 
   */
  std::vector<real_t> weight;
  /*!
   * \brief the median weights
   */
  std::vector<real_t> mid_weight;

  /*!
   * \brief the squared sum of past gradient
   */
  std::vector<real_t> squared_sum;
  /*!
   * \brief l_1 regularized coefficient
   */
  real_t l_1;
  /*!
   * \brief l_2 regularized coefficient
   */
  real_t l_2;
  /*!
   * \brief alpha for Per-coordinate learning rate
   */
  real_t alpha;
  /*!
   * \brief beta for Per-coordinate learning rate
   */
  real_t beta;
  /*!
   * \brief the total num of iterations
   */
  std::size_t niter;
  /*!
   * \brief the dimension of the weights
   */
  std::size_t dim;


};
} // namespace ftrl
