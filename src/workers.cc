
#include <dmlc/logging.h>
#include "workers.h"
#include "ftrl.h"
#include "config.h"

namespace admm{
    
Worker::Worker() {
}

Worker::Worker(const AdmmConfig& admm_params) {
    base_vec.resize(admm_params.global_weights.size());
    bias_vec.resize(admm_params.global_weights.size());
    langr_vec.resize(admm_params.global_weights.size());
    
    for(auto i = 0u; i < base_vec.size(); ++i){
        base_vec[i] = 0;
        bias_vec[i] = 0;
        langr_vec[i] = 0;
    }
}

Worker::~Worker() {
}

void Worker::Base_update(SampleSet& sample_set,const AdmmConfig& admm_params){
    FtrlConfig ftrl_params(admm_params);
    ftrl_params.l_2 = admm_params.step_size;
    ftrl_params.l_1 = 0;

    ::ftrl::FTRL_Solver ftrl_processor(ftrl_params);

    std::vector<::ftrl::FTRL_Solver::real_t> offset(ftrl_params.dim,0);
    //set the offset vector
    for(auto i = 0u;i < ftrl_params.dim; ++i){
        offset[i] = admm_params.global_weights[i] - langr_vec[i]/admm_params.step_size + bias_vec[i]; 
    }
    ftrl_processor.Run(sample_set, offset);
    
    base_vec = ftrl_processor.GetWeight();
    //recover the modified base weights 
    for(auto i = 0u;i < ftrl_params.dim; ++i){
        base_vec[i] += -admm_params.global_weights[i] + langr_vec[i]/admm_params.step_size; 
    }

}

void Worker::Bias_update(SampleSet& sample_set,const AdmmConfig& admm_params){

    FtrlConfig ftrl_params(admm_params);
    ftrl_params.l_2 = 0;
    ftrl_params.l_1 = admm_params.bias_var;

    ::ftrl::FTRL_Solver ftrl_processor(ftrl_params);

    //set the offset vector
    std::vector<::ftrl::FTRL_Solver::real_t> offset = base_vec;
    ftrl_processor.Run(sample_set, offset);

    bias_vec = ftrl_processor.GetWeight();
}

void Worker::Langrange_update(const SampleSet& sample_set,const AdmmConfig& admm_params){
    if(langr_vec.size() != admm_params.global_weights.size()){
        LOG(ERROR) << id <<"th worker: " << "langrange coefs are different with global weights in length";
        return;
    }
    for(auto i = 0u; i < langr_vec.size() ; ++i){
        langr_vec[i] += admm_params.step_size*(base_vec[i] - admm_params.global_weights[i]);
        
    }
}

const std::vector<std::pair<Worker::real_t,Worker::real_t>> Worker::GetWeights(){
    std::vector<std::pair<real_t,real_t>> pairweights;
    pairweights.resize(base_vec.size());
    for(auto i = 0u; i < base_vec.size(); ++i){
        pairweights[i] = std::pair<real_t,real_t>(base_vec[i],langr_vec[i]);
    }
    return pairweights;
}

}
