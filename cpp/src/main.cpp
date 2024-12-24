#include <torch/torch.h>
#include <H5Cpp.h>
#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

// Custom Dataset to load HDF5 data
class GoDataset : public torch::data::Dataset<GoDataset> {
public:
    GoDataset(const std::string& hdf5_file) {
        // Load HDF5 file
        H5::H5File file(hdf5_file, H5F_ACC_RDONLY);

        // Load board matrix
        H5::DataSet board_dataset = file.openDataSet("board");
        H5::DataSpace board_space = board_dataset.getSpace();
        hsize_t board_dims[3];
        board_space.getSimpleExtentDims(board_dims);
        std::vector<float> board_data(board_dims[0] * board_dims[1] * board_dims[2]);
        board_dataset.read(board_data.data(), H5::PredType::NATIVE_FLOAT);

        // Load liberty matrix
        H5::DataSet liberty_dataset = file.openDataSet("liberty");
        H5::DataSpace liberty_space = liberty_dataset.getSpace();
        hsize_t liberty_dims[3];
        liberty_space.getSimpleExtentDims(liberty_dims);
        std::vector<float> liberty_data(liberty_dims[0] * liberty_dims[1] * liberty_dims[2]);
        liberty_dataset.read(liberty_data.data(), H5::PredType::NATIVE_FLOAT);

        // Load next move
        H5::DataSet next_move_dataset = file.openDataSet("nextMove");
        H5::DataSpace next_move_space = next_move_dataset.getSpace();
        hsize_t next_move_dims[2];
        next_move_space.getSimpleExtentDims(next_move_dims);
        std::vector<int64_t> next_move_data(next_move_dims[0] * next_move_dims[1]);
        next_move_dataset.read(next_move_data.data(), H5::PredType::NATIVE_INT);

        // Populate data tensors
        int num_samples = board_dims[0];
        for (int i = 0; i < num_samples; ++i) {
            torch::Tensor board = torch::from_blob(
                board_data.data() + i * board_dims[1] * board_dims[2],
                {1, static_cast<int>(board_dims[1]), static_cast<int>(board_dims[2])});
            torch::Tensor liberty = torch::from_blob(
                liberty_data.data() + i * liberty_dims[1] * liberty_dims[2],
                {1, static_cast<int>(liberty_dims[1]), static_cast<int>(liberty_dims[2])});

            torch::Tensor input = torch::cat({board, liberty}, 0);
            data_.push_back(input);

            int row = next_move_data[i * 2];
            int col = next_move_data[i * 2 + 1];
            targets_.push_back(row * board_dims[2] + col);
        }
    }

    // Override get() to provide data samples
    torch::data::Example<> get(size_t index) override {
        return {data_[index].clone(), torch::tensor(targets_[index])};
    }

    // Override size() to return dataset size
    torch::optional<size_t> size() const override {
        return data_.size();
    }

private:
    std::vector<torch::Tensor> data_;
    std::vector<int64_t> targets_;
};

// Define GoNet: A simple convolutional neural network
struct GoNet : torch::nn::Module {
    GoNet() {
        conv1 = register_module("conv1", torch::nn::Conv2d(2, 32, 3).stride(1).padding(1));
        conv2 = register_module("conv2", torch::nn::Conv2d(32, 64, 3).stride(1).padding(1));
        conv3 = register_module("conv3", torch::nn::Conv2d(64, 128, 3).stride(1).padding(1));
        fc1 = register_module("fc1", torch::nn::Linear(128 * 19 * 19, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 19 * 19));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));
        x = torch::relu(conv2->forward(x));
        x = torch::relu(conv3->forward(x));
        x = x.view({x.size(0), -1}); // Flatten
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }

    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
};

int main() {
    // Load dataset
    auto dataset = GoDataset("./Dataset/train.h5").map(torch::data::transforms::Stack<>());
    auto dataloader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(dataset), 64);

    // Initialize the network
    auto model = std::make_shared<GoNet>();
    model->to(torch::kCUDA);

    // Define optimizer and loss function
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.01).momentum(0.9));
    torch::nn::CrossEntropyLoss loss_fn;

    // Training loop
    for (int epoch = 0; epoch < 10; ++epoch) {
        size_t batch_idx = 0;
        for (auto& batch : *dataloader) {
            auto data = batch.data.to(torch::kCUDA);
            auto target = batch.target.to(torch::kCUDA);

            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = loss_fn(output, target);
            loss.backward();
            optimizer.step();

            if (batch_idx % 10 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_idx
                          << " | Loss: " << loss.item<float>() << std::endl;
            }
            batch_idx++;
        }
    }

    // Save the trained model
    torch::save(model, "./go_model.pt");
    std::cout << "Training complete and model saved." << std::endl;

    return 0;
}
