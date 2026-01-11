# Text Generation Models

This project explores two distinct approaches to text generation: a classical Hidden Markov Model (HMM) and a modern deep learning model using Long Short-Term Memory (LSTM) networks. Each model offers unique insights into the challenges and methodologies of generating coherent and contextually relevant text. The HMM provides a probabilistic framework based on observed sequences, while the LSTM leverages its recurrent architecture to learn long-range dependencies in textual data.

## Hidden Markov Model (HMM) Text Generation

### Introduction
This section implements a basic Hidden Markov Model (HMM) for text generation. An HMM is a statistical model that outputs a sequence of observable symbols (words) based on a sequence of internal states that are not directly observable. It's particularly useful for modeling sequential data, where the probability of observing a particular item depends on the previous state.

### Features
-   **Probabilistic Word Generation**: Generates text by picking words based on the emission probabilities of the current hidden state.
-   **State Transitions**: Models the sequence of words by transitioning between hidden states based on a transition probability matrix.
-   **Customizable**: Can be adapted with different initial state probabilities, transition probabilities, and emission probabilities to generate varied text styles.

### How It Works
1.  **Initialization**: The process begins by selecting an initial hidden state based on the initial state probability distribution (pi).
2.  **Word Generation**: From the current hidden state, a word is chosen from the dictionary based on the emission probabilities (matrix B) associated with that state.
3.  **State Transition**: After generating a word, the model transitions to a new hidden state based on the transition probabilities (matrix A) from the current state.
4.  **Repetition**: Steps 2 and 3 are repeated for a specified text length, building up the generated text word by word.

### Code Explanation
-   `initialize_pos(pi)`: This function initializes the starting hidden state for the HMM based on the initial probability distribution `pi`. It uses a random number to select a state proportionally to its probability in `pi`.
-   `textGen(pi, A, B, tl, dictionary)`: This is the core text generation function for the HMM. It takes the initial state probabilities (`pi`), transition matrix (`A`), emission matrix (`B`), desired text length (`tl`), and the word dictionary as input. It iteratively generates words by first selecting a current state using `initialize_pos`, then picking a word based on the emission probabilities of that state, and finally transitioning to a new state based on the transition probabilities, repeating until the desired text length is reached.
-   `main()`: This function handles reading input for the HMM model, including the number of states (`m`), number of unique words (`n`), text length (`tl`), initial state probabilities (`pi`), transition matrix (`A`), and emission matrix (`B`). It parses these values from standard input.

### Usage
To run the HMM text generation, execute the script and provide the necessary parameters via standard input:

1.  **First Line**: `m n tl` (number of hidden states, number of unique words in dictionary, desired text length)
2.  **Second Line**: `pi` (space-separated initial state probabilities)
3.  **Next `m` Lines**: `A` (space-separated transition probabilities for each state)
4.  **Next `m` Lines**: `B` (space-separated emission probabilities for each state for each word)

### Example Input
```
2 3 5
0.5 0.5
0.7 0.3
0.4 0.6
0.1 0.4 0.5
0.6 0.3 0.1
```

### Example Output
```
0 1 0 2 1 
```
(Note: The output will consist of space-separated word indices based on the dictionary.)

## LSTM Text Generation

### Introduction
This section implements a text generation model using Long Short-Term Memory (LSTM) networks, a type of recurrent neural network (RNN) particularly well-suited for processing sequences. Unlike HMMs, LSTMs can learn long-term dependencies in data, making them highly effective for tasks like natural language processing, including text generation. The model is trained on a corpus of text to predict the next word in a sequence.

### Features
-   **Deep Learning Capabilities**: Utilizes LSTM layers to capture complex patterns and long-range dependencies in textual data.
-   **Word Embedding**: Transforms words into dense vector representations, allowing the model to understand semantic relationships.
-   **Sequence Prediction**: Predicts the next word in a sequence based on the context provided by previous words.
-   **Pre-processing Pipeline**: Includes text cleaning, tokenization, and sequence padding for robust model training.

### How It Works
1.  **Data Loading and Cleaning**: News headlines are loaded from CSV files, and then cleaned by removing punctuation and converting text to lowercase.
2.  **Tokenization and N-grams**: The cleaned text is tokenized into words, and input sequences (n-grams) are created. Each sequence consists of a series of words, where the model learns to predict the last word given the preceding ones.
3.  **Padding and Labeling**: Input sequences are padded to a uniform length. The last word of each sequence is set as the label (target word to predict), and the preceding words become the predictors.
4.  **Model Architecture**: A sequential Keras model is built with an Embedding layer, an LSTM layer, a Dropout layer to prevent overfitting, and a Dense output layer with a softmax activation function to predict word probabilities.
5.  **Training**: The model is trained on the prepared data, learning to minimize the categorical cross-entropy loss by adjusting its weights based on the provided word sequences and their next words.
6.  **Text Generation**: To generate new text, a seed text is provided. The model predicts the next word, which is then appended to the seed text, and this process is repeated to generate a longer text.

### Code Explanation
-   `clean_text(txt)`: This function preprocesses raw text by removing punctuation and converting it to lowercase, making it suitable for tokenization and model input.
-   `get_sequence_of_tokens(corpus)`: This function tokenizes the cleaned text corpus and generates input sequences (n-grams) for training. It also returns the total number of unique words in the vocabulary.
-   `generate_padded_sequences(input_sequences)`: This function takes the raw input sequences, pads them to a uniform maximum length, and separates them into predictors (input sequences) and the corresponding label (the next word to predict).
-   `create_model(max_sequence_len, total_words)`: This function defines and constructs the LSTM neural network model. It includes an Embedding layer, an LSTM layer, a Dropout layer, and a Dense output layer with softmax activation.
-   `model.fit(predictors, label, epochs=100, verbose=5)`: This line initiates the training process of the LSTM model using the prepared predictors and labels. It trains for 100 epochs with verbose output.
-   `generate_text(seed_text, next_words, model, max_sequence_len)`: This function is used to generate new text. It takes a starting `seed_text`, the number of `next_words` to generate, the trained `model`, and the `max_sequence_len` to format inputs correctly. It iteratively predicts and appends words to the seed text.

### Usage
To generate text using the LSTM model:
1.  Ensure all necessary Python packages are installed (see Prerequisites/Installation).
2.  Run the script. It will load data, preprocess it, train the model, and then generate an example text.
3.  Modify the `generate_text` call at the end of the script with your desired `seed_text` and `next_words` to generate different outputs.

### Example Output
```
India And China To Work Out Details Of India Pact
```
(Note: The generated text will vary each time the model is trained due to the stochastic nature of training and data loading from multiple files, even with random seeds set.)

## Prerequisites/Installation

To run these text generation models, you will need to have Python installed (preferably Python 3.7+). Additionally, the following Python libraries are required:

-   `numpy`: Fundamental package for scientific computing with Python.
-   `pandas`: Data structures and data analysis tools.
-   `keras`: High-level neural networks API (often included with TensorFlow).
-   `tensorflow`: Open-source machine learning framework (Keras is integrated into TensorFlow 2.x).

You can install these packages using pip:

```bash
pip install numpy pandas tensorflow keras
```

**Note on TensorFlow and Keras:**
For TensorFlow 2.x, Keras is integrated as `tf.keras`. If you encounter compatibility issues, you might need to specify an older version of TensorFlow (e.g., `tensorflow==2.x.x`) or ensure your Keras installation is compatible with your TensorFlow version.
