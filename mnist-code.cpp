/* Sample code for training a FCN on MNIST dataset using PyTorch C++ API */
/* This code uses VGG-16 Layer Network */
/* Dataset link: http://yann.lecun.com/exdb/mnist/ */

#include "main.h"

// constexpr: constant expression, applied to variables, funtions & class constructors
// train size, test size
constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
// 
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
// size- (28, 28)
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
// setting filemanes
// Note that filenames by default is set as "train-images-idx3-ubyte" and downloaded file will be of "train-images.idx3-ubyte" kind. Do correct them before proceeing! (although this point not needed is here)
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

using namespace std;
using namespace torch;

// constructs an object of type Net, returns shared_ptr: that owns and stores pointer to it
auto net = std::make_shared<Net>();

std::string root_string = "/Users/krshrimali/Documents/krshrimali-blogs/bhaiya/fashion-mnist/";

// CustomDataset(): creates dataset as per needs of users
// .map(): for transformations
// torch::data::transforms::Stack<>(): create stack of image and label
// make_data_loader: for loading data
// torch::data::samplers::RandomSampler: randomly takes images and their labels from sample
auto train_dataset = CustomDataset(root_string, true).map(torch::data::transforms::Stack<>());
auto train_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_dataset), 32);

auto test_dataset = CustomDataset(root_string, false).map(torch::data::transforms::Stack<>());
auto test_data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), 4);

// optimizer: set as SGD
// Learning Rate 0.01
torch::optim::SGD optimizer(net->parameters(), 0.01); 

Tensor process_images(const std::string& root, bool train) {
  /* Function to process image
  
  Parameters
  ==========
  root: type- string
        absolute path of file
  train: bool
         wheater the given folder is training or testing
         
   Results
   =======
   vector of tensor (images)
   */
  
  // see main.h function
  // it joins path of root (here, "/Users/krshrimali/Documents/krshrimali-blogs/bhaiya/fashion-mnist/") and train ("train-images-idx3-ubyte" or "t10k-images-idx3-ubyte") as per input passed
  // it checks if '/' is present in adddress or not, if not adds it and then joins path
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  // ifstream: to operate in files
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  // see main.h function
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  // torch::empty : returns tensor filled with uninitialized data
  // reinterpret_cast: casting operator, convert one pointer to another pointer 
  // numel(): returns total number of elements in input tensor
  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

// function to process labels
Tensor process_labels(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainTargetsFilename : kTestTargetsFilename);
  std::ifstream targets(path, std::ios::binary);
  TORCH_CHECK(targets, "Error opening targets file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  expect_int32(targets, kTargetMagicNumber);
  expect_int32(targets, count);

  auto tensor = torch::empty(count, torch::kByte);
  targets.read(reinterpret_cast<char*>(tensor.data_ptr()), count);
  return tensor.to(torch::kInt64);
}

void train(torch::optim::Optimizer& optimizer, size_t dataset_size, int epoch) {
  /*
  This function trains the network on our data loader using optimizer for given number of epochs.

  Parameters
  ==========
  torch::optim::Optimizer& optimizer: Optimizer like Adam, SGD etc.
  size_t dataset_size: Size of training dataset
  int epoch: Number of epoch for training
  
  Returns
  =======
  Nothing (void)
  */

  net->train();
  
  // initialization 
  size_t batch_index = 0;
  float mse = 0;
  float Acc = 0.0;
  
  for(auto& batch: *train_data_loader) {
    auto data = batch.data;
    auto target = batch.target.squeeze();          
    
    // Should be of length: batch_size
    data = data.to(torch::kF32);
    
    // coverting traget to type kInt64
    target = target.to(torch::kInt64);

    // reset gradients
    // used: to set gradients to zero before starting backpropagation
    optimizer.zero_grad();

    // forward propagation
    auto output = net->forward(data, data.sizes()[0]);
    // nll_loss: negative log likelihood loss
    auto loss = torch::nll_loss(output, target);
    
    // computing gradients 
    loss.backward();
    
    // updating parameters 
    optimizer.step();

    auto acc = output.argmax(1).eq(target).sum();
    Acc += acc.template item<float>();
    mse += loss.template item<float>();

    batch_index += 1;
  }

  // Take mean of loss
  mse = mse/float(batch_index);
  
  std::cout << "Epoch: " << epoch << ", " << "Accuracy: " << Acc/dataset_size << ", " << "MSE: " << mse << std::endl;
  torch::save(net, "best_model_try.pt");
}

void test(size_t data_size) {
    /*
     Function to test the network on test data
     
     Parameters
     ===========
     1. data_size (size_t type) - test data size
     
     Returns
     =======
     Nothing (void)
     */
    net->eval();
    
    float Loss = 0, Acc = 0;
    
    for (const auto& batch : *test_data_loader) {
        auto data = batch.data.view({4, -1});
        auto targets = batch.target.view({-1});
        
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);
        
        auto output = net->forward(data, data.sizes()[0]);

        // output = output.view({output.size(0), -1});
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    
    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main() {
  // training 
  for(int epoch=0; epoch<10; epoch++)
    train(optimizer, train_dataset.size().value(), epoch);
  
  // testing 
  test(test_dataset.size().value());
}
