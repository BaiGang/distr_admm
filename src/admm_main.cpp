
#include <rabit.h>
#include <dmlc/logging.h>

int main(int argc, char* argv[]) {
  ::dmlc::InitLogging(argv[0]);
  
  ::rabit::Init(argc, argv);

  ::rabit::Finalize();
  
  return 0;
}

