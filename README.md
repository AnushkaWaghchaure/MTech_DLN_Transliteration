MTech_DLN_Transliteration
Character-level Hindi Transliteration using RNN, LSTM, GRU and Transformer


## Project Overview

This project is part of my **M.Tech AI-ML lab assignment**. It focuses on character-level transliteration from Romanized Hindi (English letters) into Devanagari Hindi.

For example:

| Romanized Input | Hindi Output |
|---|---|
| sikhaaega | सिखाएगा |
| kolhapur | कोल्हापुर |
| tirunelveli | तिरुनेलवेली |

The project explores different deep learning architectures, including RNN, LSTM, GRU, Attention-based Seq2Seq and Transformer models.

## Objectives

- Understand sequence-to-sequence learning using recurrent neural networks.
- Compare RNN, LSTM and GRU architectures.
- Implement an encoder-decoder model.
- Understand and implement an attention mechanism.
- Visualize attention weights using heatmaps.
- Implement Transformer components manually.
- Compare model performance using test accuracy.
- Perform hyperparameter tuning using Weights & Biases (W&B).

## Dataset

This project uses the **Hindi (hin) subset of the Aksharantar Sampled Dataset** provided for the assignment.

### Dataset Distribution

| Dataset | Number of Samples |

| Training | 51,200 |
| Validation | 4,096 |
| Testing | 4,096 |
| **Total** | **59,392** |

### Preprocessing

- Character-level tokenization.
- Input vocabulary size: 30.
- Target vocabulary size: 68.
- Special tokens: PAD, SOS, EOS and UNK.
- Input sequence length: 24.
- Target sequence length: 21.
- Padding applied to maintain consistent sequence lengths.

## Models Implemented

### 1. Seq2Seq Model

A configurable encoder-decoder architecture was implemented using:

- RNN
- LSTM
- GRU
- Character embeddings
- Hidden layers
- Bidirectional encoder
- Dropout

The model supports different configurations of embedding size, hidden size and recurrent layers.

The theoretical parameter count for the selected configuration was **180,548**, matching the actual PyTorch parameter count.

### 2. W&B Hyperparameter Sweep

A random hyperparameter sweep was performed using Weights & Biases.

The explored hyperparameters included:

- Cell type
- Embedding size
- Hidden size
- Encoder layers
- Decoder layers
- Bidirectional setting
- Dropout
- Learning rate

Five sweep runs were completed.

The W&B dashboard was used to visualize training loss, validation loss, validation accuracy, parameter importance and parallel coordinates.

### 3. Attention-Based Seq2Seq

An attention mechanism was added to the encoder-decoder model.

It allows the decoder to focus on different input character positions while generating the output sequence.

Attention heatmaps were generated for ten test examples to visualize the alignment between input and output characters.

### 4. Manual Transformer Implementation

The Transformer was implemented manually without using `torch.nn.Transformer`.

The implementation includes:

- Positional Encoding
- Multi-Head Attention
- Encoder Block
- Decoder Block
- Causal Masking
- Padding Masking
- Feed-Forward Network
- Residual Connections
- Batch Normalization

#### Transformer Configuration

| Parameter | Value |
|---|---:|
| Embedding Size | 64 |
| Attention Heads | 4 |
| Encoder Layers | 1 |
| Decoder Layers | 1 |
| Feed-Forward Size | 256 |
| Batch Normalization | True |
| Total Parameters | 127,428 |

## Model Evaluation

All models were evaluated on the same held-out test dataset containing 4,096 samples.

### Test Results

| Model              | Correct Predictions | Test Samples | Exact-Match Accuracy |

| Vanilla Seq2Seq    | 966                 | 4,096        | 23.58% |
| Attention Seq2Seq  | 1,479               | 4,096        | 36.11% |
| Transformer        | 1,214               | 4,096        | 29.64% |

### Observations

- The vanilla Seq2Seq model achieved 23.58% exact-match accuracy.
- The attention-based model achieved the highest accuracy of 36.11%.
- The Transformer achieved 29.64% accuracy.
- Attention improved the performance compared with the vanilla model in these experiments.

The attention-based Seq2Seq model performed best among the three evaluated models.

**Note:** Exact-match accuracy considers a prediction correct only when the complete predicted word matches the actual Hindi word.

## Results and Visualizations

The project includes:

- W&B training and validation graphs.
- Validation accuracy versus creation-time graph.
- Hyperparameter importance visualization.
- Parallel coordinates plot.
- Attention heatmaps for nine test inputs.
- A separate attention heatmap for the tenth input.
- CSV files containing actual and predicted transliterations.


## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Weights & Biases
- Google Colab
- GitHub

## How to Run the Project

1. Open the notebook from this GitHub repository.
2. Click **Open in Colab** or download the notebook and upload it to Google Colab.
3. Upload or make available the instructor-provided Aksharantar dataset ZIP file.
4. Check that the dataset path in the notebook is correct.
5. Select **Runtime → Change runtime type → T4 GPU**, if available.
6. Execute the notebook cells in order.
7. Observe the training, validation and test results.

### Install Required Libraries

Run this command in Google Colab if the required libraries are not already installed:

```python
!pip install torch numpy pandas matplotlib scikit-learn wandb
```

## Limitations

- The results depend on the dataset, model configuration and training settings.
- Only five runs were performed in the W&B sweep.
- The Transformer was trained using one configuration.
- The attention model performed better than the Transformer in this particular experiment.
- Further experiments with more training epochs and hyperparameter configurations may improve performance.

## Conclusion

Through this project, I gained practical experience in character-level sequence-to-sequence learning and neural network architectures.

I implemented and evaluated recurrent models, added an attention mechanism, visualized attention weights and built a Transformer using its main components.

The experimental results showed that the attention-based Seq2Seq model achieved the highest test accuracy of 36.11%, followed by the Transformer at 29.64% and vanilla Seq2Seq at 23.58%.

Overall, this assignment helped me understand how different deep learning architectures handle sequence-to-sequence transliteration tasks.

## Project Links

**GitHub Repository:**

https://github.com/AnushkaWaghchaure/MTech_DLN_Transliteration

**Weights & Biases Sweep:**

https://wandb.ai/anushka_waghchaure-mit-world-peace-university/MTech_AI_ML_Transliteration/sweeps/llpulyx5


