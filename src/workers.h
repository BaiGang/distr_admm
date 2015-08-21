#pragma once
#include <vector>
#include <dmlc/logging.h>
#include <dmlc/data.h>
#include "sample_set.h"
#include "config.h"

namespace  admm{

/*!
 * \brief Worker executing single logistic regression update
 * 
 * \param w_vec the weight vector for the task
 * \param v_vec the offset vector for the task
 * \param alpha_vec the lagrange efficient vector for the task
*/
class  Worker {

   public:
    typedef float real_t;
    
    Worker();
    Worker(const AdmmConfig& admm_params);
    virtual ~Worker();
    
    /*!
     * \brief update the base weights of the single model.
     * \param batches the traning instances specified by the task 
     * \param params the configure parameters
     */
    void Base_update(SampleSet& sample_set,const AdmmConfig& admm_params);

    /*!
     * \brief update the bias weights of the single model.
     */
    void Bias_update(SampleSet& sample_set,const AdmmConfig& admm_params);

    /*!
     * \brief update the langrange coefficients of the single model.
     */
    void Langrange_update(const SampleSet& sample_set,const AdmmConfig& admm_params);

    
    /*!
     * \brief return the final base weights and langranges of the single model
     */
    const std::vector<std::pair<real_t,real_t>> GetWeights();
   private:
    
    /*!
     * \brief worker's ID
     */
    std::size_t id;

    /*!
     * \brief the base weights
     */
    std::vector<real_t> base_vec;
    /*!
     * \brief the bias weights
     */
    std::vector<real_t> bias_vec;
    /*!
     * \brief the langrange coefficients
     */
    std::vector<real_t> langr_vec;

};

}

