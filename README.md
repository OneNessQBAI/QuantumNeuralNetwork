# Quantum Neural Network (QNN)

This project implements a Quantum Neural Network that integrates classical neural network models with quantum computing algorithms to enhance data processing capabilities.

## Project Structure

```
QNN/
├── main_workflow.py
├── NomicEmbed.py
├── Qwen.py
├── Qwen2Math.py
├── DeepSeekCoder.py
├── Starcoder.py
├── system_prompts/
│   ├── nomic_embed_prompt.txt
│   ├── qwen_prompt.txt
│   ├── qwen2_math_prompt.txt
│   ├── tocoder_prompt.txt
│   └── starcoder_prompt.txt
├── synthetic_data_layers/
│   ├── data_layer_nomic_qwen.py
│   ├── data_layer_qwen_qwen2_math.py
│   ├── data_layer_qwen2_math_tocoder.py
│   └── data_layer_tocoder_starcoder.py
├── quantum_algorithms/
│   └── quantum_optimization.py
└── [Various quantum algorithm implementation files]

```

## Components

1. **Main Workflow**: `main_workflow.py` orchestrates the entire QNN process.
2. **Models**: NomicEmbed, Qwen, Qwen2Math, DeepSeekCoder, and Starcoder.
3. **System Prompts**: Contains prompts for each model.
4. **Synthetic Data Layers**: Manages data transformation between models.
5. **Quantum Algorithms**: Implements quantum optimization techniques.

## How to Run

1. Ensure you have Python 3.7+ installed.
2. Install required packages:
   ```
   pip install cirq requests beautifulsoup4
   ```
3. Navigate to the project directory:
   ```
   cd path/to/QNN
   ```
4. Run the main workflow:
   ```
   python main_workflow.py
   ```

## Testing

The `main_workflow.py` script includes a set of test questions covering various domains. These questions are processed through the QNN system, demonstrating its capabilities in handling complex queries, coding tasks, and quantum-related problems.

## Customization

To extend or modify the QNN:

1. Add new models in the root directory.
2. Create corresponding system prompts in the `system_prompts` directory.
3. Implement new data layers in the `synthetic_data_layers` directory.
4. Extend quantum algorithms in the `quantum_algorithms` directory.
5. Update `main_workflow.py` to incorporate new components.

## Note

This project is a demonstration of integrating quantum computing concepts with neural networks. It's designed for educational and experimental purposes.

For any questions or contributions, please contact the project maintainer.