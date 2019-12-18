/* Sample code for training a FCN on MNIST dataset using PyTorch C++ API */
/* This code uses VGG-16 Layer Network */

#include "main.h"

constexpr uint32_t kTrainSize = 60000;
constexpr uint32_t kTestSize = 10000;
constexpr uint32_t kImageMagicNumber = 2051;
constexpr uint32_t kTargetMagicNumber = 2049;
constexpr uint32_t kImageRows = 28;
constexpr uint32_t kImageColumns = 28;
constexpr const char* kTrainImagesFilename = "train-images-idx3-ubyte";
constexpr const char* kTrainTargetsFilename = "train-labels-idx1-ubyte";
constexpr const char* kTestImagesFilename = "t10k-images-idx3-ubyte";
constexpr const char* kTestTargetsFilename = "t10k-labels-idx1-ubyte";

using namespace std;
using namespace torch;

auto net = std::make_shared<Net>();

Tensor process_images(const std::string& root, bool train) {
  const auto path =
      join_paths(root, train ? kTrainImagesFilename : kTestImagesFilename);
  std::ifstream images(path, std::ios::binary);
  TORCH_CHECK(images, "Error opening images file at ", path);

  const auto count = train ? kTrainSize : kTestSize;

  // From http://yann.lecun.com/exdb/mnist/
  expect_int32(images, kImageMagicNumber);
  expect_int32(images, count);
  expect_int32(images, kImageRows);
  expect_int32(images, kImageColumns);

  auto tensor =
      torch::empty({count, 1, kImageRows, kImageColumns}, torch::kByte);
  images.read(reinterpret_cast<char*>(tensor.data_ptr()), tensor.numel());
  return tensor.to(torch::kFloat32).div_(255);
}

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

template<typename Dataloader>
void train(Net& net, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size, int epoch) {
  /*
  This function trains the network on our data loader using optimizer for given number of epochs.

  Parameters
  ==================
  ConvNet& net: Network struct
  DataLoader& data_loader: Training data loader
  torch::optim::Optimizer& optimizer: Optimizer like Adam, SGD etc.
  size_t dataset_size: Size of training dataset
  int epoch: Number of epoch for training
  */

  net.train();
  
  size_t batch_index = 0;
  float mse = 0;
  float Acc = 0.0;

  for(auto& batch: *data_loader) {
    auto data = batch.data;
    auto target = batch.target.squeeze();
    
    // Should be of length: batch_size
    data = data.to(torch::kF32);
    target = target.to(torch::kInt64);

    optimizer.zero_grad();

    auto output = net.forward(data);
    auto loss = torch::nll_loss(output, target);

    loss.backward();
    optimizer.step();

    auto acc = output.argmax(1).eq(target).sum();
    Acc += acc.template item<float>();
    mse += loss.template item<float>();

    batch_index += 1;
  }

  mse = mse/float(batch_index); // Take mean of loss

  std::cout << "Epoch: " << epoch << ", " << "Accuracy: " << Acc/dataset_size << ", " << "MSE: " << mse << std::endl;
  torch::save(net, "best_model_try.pt");
}

template<typename Dataloader>
void test(Net& network, Dataloader& loader, size_t data_size) {
    /*
     Function to test the network on test data
     
     Parameters
     ===========
     1. network (torch::jit::script::Module type) - Pre-trained model without last FC layer
     2. loader (Dataloader& type) - test data loader
     3. data_size (size_t type) - test data size
     
     Returns
     ===========
     Nothing (void)
     */
    network.eval();
    
    float Loss = 0, Acc = 0;
    
    for (const auto& batch : *loader) {
        auto data = batch.data;
        auto targets = batch.target.view({-1});
        
        data = data.to(torch::kF32);
        targets = targets.to(torch::kInt64);

        auto output = network.forward(data);
        
        // output = output.view({output.size(0), -1});
        auto loss = torch::nll_loss(torch::log_softmax(output, 1), targets);
        auto acc = output.argmax(1).eq(targets).sum();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();
    }
    
    std::cout << "Test Loss: " << Loss/data_size << ", Acc:" << Acc/data_size << std::endl;
}

int main() {
	std::string root_string = "/Users/krshrimali/Documents/krshrimali-blogs/bhaiya/fashion-mnist/";
	bool train_ = true;
	auto custom_dataset = CustomDataset(root_string, train_).map(torch::data::transforms::Stack<>());

	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), 32);
  train_ = false;
  auto test_dataset = CustomDataset(root_string, train_).map(torch::data::transforms::Stack<>());
	auto test_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(test_dataset), 32);
  // // Create multi-threaded data loader for MNIST data
	// auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	// 		std::move(torch::data::datasets::MNIST("/absolute/path/to/data").map(torch::data::transforms::Normalize<>(0.13707, 0.3081)).map(
	// 			torch::data::transforms::Stack<>())), 64);
	
  // Build VGG-16 Network

  torch::optim::SGD optimizer(net->parameters(), 0.01); // Learning Rate 0.01

  train(data_loader, optimizer, custom_dataset.size().value());

  // train(Net& net, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size, int epoch)
  test(test_loader, test_dataset.size().value());

	// // net.train();
 //  float Acc = 0.0;
	// for(size_t epoch=1; epoch<=10; ++epoch) {
	// 	size_t batch_index = 0;
	// 	// Iterate data loader to yield batches from the dataset
	// 	for (auto& batch: *data_loader) {
 //      auto target = batch.target.squeeze();
 //      target = target.to(torch::kInt64);

 //      auto data = batch.data;
 //      data = data.to(torch::kF32);

	// 		// Reset gradients
	// 		optimizer.zero_grad();
	// 		// Execute the model
	// 		torch::Tensor prediction = net->forward(data);
	// 		// Compute loss value
	// 		torch::Tensor loss = torch::nll_loss(prediction, target);
	// 		// Compute gradients
	// 		loss.backward();
	// 		// Update the parameters
	// 		optimizer.step();

 //      auto acc = prediction.argmax(1).eq(target).sum().item<float>();
 //      Acc += acc;

	// 		// Output the loss and checkpoint every 100 batches
	// 		if (++batch_index % 10 == 0) {
	// 			std::cout << "Epoch: " << epoch << " | Batch: " << batch_index 
	// 				<< " | Loss: " << loss.item<float>() << " | Acc: " << acc/32 << std::endl;
	// 		}
	// 	}

 //    std::cout << "Epoch: " << epoch << ", Accuracy: " << Acc/60000 << std::endl;
 //    torch::save(net, "net.pt");
	// }
}