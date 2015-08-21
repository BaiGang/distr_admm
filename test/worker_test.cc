
#include <vector>
#include <dmlc/data.h>
#include "gtest.h"
#include "../src/workers.h"
#include "../src/master.h"
#include "../src/config.h"

namespace admm{
    
class WorkerTest: public ::testing::Test{
 public:
  virtual void SetUp();
  virtual void TearDown();

 protected:
};

void WorkerTest::SetUp(){
}

void WorkerTest::TearDown(){
}

TEST_F(WorkerTest, update){
   typedef float real_t;

   SampleSet sample_set;
   sample_set.Initialize("test/wtest.libsvm",0,1);

   AdmmConfig admm_params(1,1,1,5); 

   Worker Worker_processor(admm_params);
   Master Master_processor;

   //update of the local base and bias weights
   Worker_processor.Bias_update(sample_set, admm_params);
   Worker_processor.Base_update(sample_set, admm_params);

   //test the local base weights
   std::vector<std::pair<real_t,real_t>> result = Worker_processor.GetWeights();
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
        EXPECT_DOUBLE_EQ(0,result[i].first);

   //update of the global weights
   std::vector<std::vector<std::pair<real_t,real_t>>> RESULT;
   RESULT.push_back(result);
   Master_processor.global_update(RESULT, admm_params);
   //test the global weights
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
        EXPECT_DOUBLE_EQ(0, admm_params.global_weights[i]);

   //update of the langranges
   Worker_processor.Langrange_update(sample_set, admm_params);
   //test the langranges
   result = Worker_processor.GetWeights();
   //TODO
   for(auto i = 0u; i < admm_params.global_weights.size(); ++i)
   EXPECT_DOUBLE_EQ(0,result[i].second);
}
}
