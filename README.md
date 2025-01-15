![CNN](images/CNN.png)

# AlphaGo-Lite

**AlphaGo-Lite** is a simplified implementation of an AI Go player inspired by AlphaGo. It leverages a deep learning-based approach for predicting moves on a 19x19 Go board. This project combines ONNX-based inference with custom Go game logic to provide a lightweight yet powerful Go-playing AI engine.

## Features

- **Deep Learning Inference**: Supports ONNX models for efficient move prediction.
- **Customizable Game Core**: Built on a modular GTP core for Go game logic.
- **Training Pipeline**: Includes scripts for preprocessing SGF files, training neural networks, and exporting ONNX models.
- **Flexible File Support**: Handles HDF5 datasets and exports Go positions as `.npy` files.
- **Interactive Gameplay**: Play against the AI with real-time predictions.

## Installation

### Clone the Repository
```bash
# Clone the repository
git clone https://github.com/yourusername/AlphaGo-Lite.git
cd AlphaGo-Lite
```

### Install Dependencies

#### Install Python Dependencies

1. [Install PyTorch](https://pytorch.org/get-started/locally/)

2. Install other Python dependencies:
```bash
pip install h5py matplotlib numpy
```

#### Build the Game Core
1. Navigate to the `GoGame-Core` submodule:
   ```bash
   cd GoGame-Core
   ```
2. Build the C++ core:
   ```bash
   cd src
   mkdir build && cd build
   cmake ..
   make
   ```

3. Verify the core:
   ```bash
   ./GoGame-Core
   ```
4. Navigate back to the root directory:
   ```bash
   cd ../../
   ```

### Setting Up Datasets

1. I recommend: [Joe's Go Datebase](https://pjreddie.com/projects/jgdb/)
2. If you downloaded the dataset from recommended site, un-zip the downloaded file under root directory and change directory name to "rawData". There are about 500,000 files, organized in sub-folder, each containing 1000 files.
3. If you prepared your own dataset, place your SGF datasets in the `RawData/sgf` directory and divide your dataset to "RawData/sgf/train" and "RawData/sgf/test" 
4. Use the provided Bash scripts to preprocess and merge datasets:
   ```bash
   ./processSGF.sh process
   ./processSGF.sh merge train
   ./processSGF.sh merge test
   ```

## Usage

### Train the Model

#### Phase 1: Supervised Training
1. Run the training script:
   ```bash
   python python/TrainCNN.py
   ```
   or more safely:
   ```bash
   nohup python python/TrainCNN.py
   ```

2. Monitor the accuracy and loss as the model trains. The CNN should be able to achieve 32% under the recommended dataset.

#### Phase 2: Reinforcement Training

Working on this...

#### Phase 3: Value Network

Working on this...

### Export the Model

Convert the trained PyTorch model to ONNX:
```bash
python python/export_onnx.py
```

## File Structure

```plaintext
AlphaGo-Lite/
├── GoGame-Core          # Board-keeping core, written in C++
├── cpp                  # C++ Module, Python is too slow
├── python               # Everything about the CNN
├── processSGF.sh        # RawData -> Dataset magic
├── RawData              # SGF files and processed HDF5 datasets
│   ├── sgf
│   │   ├── test
│   │   ├── train
│   │   └── val
├── Dataset              # Merged Dataset - Ready to be loaded
│   ├── test             # Test set
│   ├── train            # Train set
└── README.md
```

## Contributing

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
4. Push to your branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by [DeepMind's AlphaGo](https://deepmind.com/research/highlighted-research/alphago).
- Built with love for the Go community.

---

Enjoy playing and improving your Go skills with **AlphaGo-Lite**! Feel free to reach out with suggestions or contributions!
