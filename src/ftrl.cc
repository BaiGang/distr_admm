
#include <stdlib.h>
#include <algorithm>
#include <dmlc/logging.h>
#include "ftrl.h"


namespace ftrl{

FTRL_Solver::FTRL_Solver(FTRL_Solver::real_t lambda_1,
						 FTRL_Solver::real_t lambda_2,
						 FTRL_Solver::real_t alpha_init,
						 FTRL_Solver::real_t beta_init,
						 std::size_t niter_init,
						 std::size_t dim_init): l_1(lambda_1), l_2(lambda_2), alpha(alpha_init), beta(beta_init), niter(niter_init), dim(dim_init) {

	weight.resize(dim);
	mid_weight.resize(dim);
	squared_sum.resize(dim);

	for(auto i = 0u; i < dim; ++i){
		weight[i] = 0;
		mid_weight[i] = 0;
		squared_sum[i] = 0;
	}
} 

FTRL_Solver::FTRL_Solver(::admm::FtrlConfig& params) {
    l_1 = params.l_1;
    l_2 = params.l_2;
    alpha = params.alpha;
    beta = params.beta;
    dim = params.dim;
    niter = params.niter;

	weight.resize(dim);
	mid_weight.resize(dim);
	squared_sum.resize(dim);

	for(auto i = 0u; i < dim; ++i){
		weight[i] = 0;
		mid_weight[i] = 0;
		squared_sum[i] = 0;
	}
}


FTRL_Solver::~FTRL_Solver(){
}

FTRL_Solver::real_t FTRL_Solver::Predict(FTRL_Solver::Row& x) {
	
	for(auto i = 0u; i < x.length; ++i){
		// w[i] =		 0													 if |z[i]| <= l_1
		//		  (sgn(z[i])*l_1 - z[i])/((beta + sqrt(n[i]))/alpha + l_2)   otherwise.
		//
		real_t sign = mid_weight[x.index[i]] < 0? -1:1;
		weight[x.index[i]] = abs(mid_weight[x.index[i]]) <= l_1? 0:(sign*l_1 - mid_weight[x.index[i]])/((beta + sqrt(squared_sum[x.index[i]]))/alpha + l_2);
	}

	real_t* ptr_weight = &weight[0];

	auto inner_product = x.SDot(ptr_weight,dim);

    // P(y=1|x,w) = 1/(1 + exp(-<w,x>)) 
	return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

FTRL_Solver::real_t FTRL_Solver::Predict(FTRL_Solver::Row& x, std::vector<FTRL_Solver::real_t>& offset) {
    for(auto i = 0u; i < x.length; ++i){
        LOG(INFO) << "x.index[" << i << "] = " << x.index[i] << "\n";
		real_t sign = mid_weight[x.index[i]] < 0? -1:1;
		weight[x.index[i]] = abs(mid_weight[x.index[i]]) <= l_1? 0:(sign*l_1 - mid_weight[x.index[i]])/((beta + sqrt(squared_sum[x.index[i]]))/alpha + l_2);
    }

    real_t* ptr_weight = &weight[0];
    real_t* ptr_offset = &offset[0];

    auto inner_product = x.SDot(ptr_weight,dim) + x.SDot(ptr_offset,dim);

	return 1.0/(1 + exp(- std::max(std::min(inner_product, (float)35), (float)(-35)))); 
}

void FTRL_Solver::Update(const FTRL_Solver::Row& x, FTRL_Solver::real_t predict) {

	for(auto i = 0u; i < x.length; ++i){
	    //g[i] = (p - y)*x[i]
	    auto loss = (predict - x.label) * x.value[i];
		auto sigma = (sqrt(squared_sum[x.index[i]] + loss*loss) - sqrt(squared_sum[x.index[i]]))/alpha;
		//z[i] = z[i] + g[i] - sigma*w[i]
		mid_weight[x.index[i]] += loss - sigma * weight[x.index[i]];
		//n[i] = n[i] + g[i]^2;
		squared_sum[x.index[i]] += loss * loss;
	}
}


void FTRL_Solver::Run(FTRL_Solver::SampleSet& sample_set, std::vector<FTRL_Solver::real_t>& offset) {
	
	for(auto i = 0u; i < niter ; ++i){
		sample_set.Rewind();
		while(sample_set.Next()){
			Row x = sample_set.GetData();
            if(offset.size() == 0){
			    auto predict = Predict(x);
			    Update(x,predict);
            }
            else{
			    auto predict = Predict(x,offset);
			    Update(x,predict);
            }
		}
	}
}

const std::vector<FTRL_Solver::real_t>& FTRL_Solver::GetWeight() {
	return weight;
}

void FTRL_Solver::Assign(const std::vector<FTRL_Solver::real_t>& x){
    dim = x.size();
    weight.resize(dim);
    LOG(INFO) << "The dim of weight is " << dim << " now in ftrl process." << "\n";
    for(auto i = 0u;  i < x.size(); ++i){
        weight[i] = x[i];
    }
}

} // namespace ftrl
