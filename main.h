//
//  main.h
//  transfer-learning
//
//  Created by Kushashwa Ravi Shrimali on 15/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#ifndef main_h
#define main_h

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <dirent.h>
#include <torch/script.h>

struct Net: public torch::nn::Module {
    // VGG-16 Layer
    // conv1_1 - conv1_2 - pool 1 - conv2_1 - conv2_2 - pool 2 - conv3_1 - conv3_2 - conv3_3 - pool 3 -
    // conv4_1 - conv4_2 - conv4_3 - pool 4 - conv5_1 - conv5_2 - conv5_3 - pool 5 - fc6 - fc7 - fc8
    
    // Note: pool 5 not implemented as no need for MNIST dataset
    Net() {
        fc1 = register_module("fc1", torch::nn::Linear(112*7, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 512));
        fc3 = register_module("fc3", torch::nn::Linear(512, 10));
    }

    // Implement Algorithm
    torch::Tensor forward(torch::Tensor x, int batch_size) {
        x = x.view({batch_size, -1});
        x = torch::relu(fc1->forward(x));
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return torch::log_softmax(x, 1);
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

bool check_is_little_endian() {
  const uint32_t word = 1;
  return reinterpret_cast<const uint8_t*>(&word)[0] == 1;
}

constexpr uint32_t flip_endianness(uint32_t value) {
  return ((value & 0xffu) << 24u) | ((value & 0xff00u) << 8u) |
      ((value & 0xff0000u) >> 8u) | ((value & 0xff000000u) >> 24u);
}

uint32_t read_int32(std::ifstream& stream) {
  static const bool is_little_endian = check_is_little_endian();
  uint32_t value;
  AT_ASSERT(stream.read(reinterpret_cast<char*>(&value), sizeof value));
  return is_little_endian ? flip_endianness(value) : value;
}

uint32_t expect_int32(std::ifstream& stream, uint32_t expected) {
  const auto value = read_int32(stream);
  // clang-format off
  TORCH_CHECK(value == expected,
      "Expected to read number ", expected, " but found ", value, " instead");
  // clang-format on
  return value;
}

std::string join_paths(std::string head, const std::string& tail) {
  if (head.back() != '/') {
    head.push_back('/');
  }
  head += tail;
  return head;
}

// Function returns vector of tensors (images) read from the list of images in a folder
torch::Tensor process_images(const std::string& root, bool train);

// Function returns vector of tensors (labels) read from the list of labels
torch::Tensor process_labels(const std::string& root, bool train);

// Function to load data from given folder(s) name(s) (folders_name)
// Returns pair of vectors of string (image locations) and int (respective labels)
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

// Function to train the network on train data
void train(torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
void test(size_t data_size);

// Custom Dataset class
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    torch::Tensor states, labels;
    size_t ds_size;
public:
    CustomDataset(const std::string& root, bool train) {
        states = process_images(root, train);
        labels = process_labels(root, train);
        ds_size = states.size(0);
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states[index];
        torch::Tensor sample_label = labels[index];
        return {sample_img.clone(), sample_label.clone()};
    };
    
    torch::optional<size_t> size() const override {
        return ds_size;
    };
};

#endif /* main_h */