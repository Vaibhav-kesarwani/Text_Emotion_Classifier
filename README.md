# Text Emotion Classifier 

## Project Overview
The **Text Emotion Classifier** is a machine learning project that aims to detect and classify emotions in text data. The model uses deep learning techniques to process textual input and identify the underlying emotion, such as happiness, sadness, anger, or surprise. This project can be used for sentiment analysis, social media monitoring, customer feedback processing, and more.

## Dataset
The dataset used in this project is a labeled text dataset where each entry consists of a text sample and its corresponding emotion label. The dataset is stored in a `train.txt` file with the following structure:

- **Text:** The input text string.
- **Emotions:** The associated emotion label (e.g., "happy", "sad", "angry", etc).
- **Dataset link:** [Kaggle link](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)

An example of the data structure:
```text
I am feeling great today!;happy
This is so frustrating and annoying.;angry
```

The data file `train.txt` should be placed in the root directory of the project.

## Installation

To run this project locally, you'll need to have Python installed. Follow these steps to set up the environment:

1. **Clone the Repository:**
```bash
git clone https://github.com/Vaibhav-kesarwani/Text_Emotion_Classifier.git
cd Text_Emotion_Classifier
```

2. **Create a Virtual Environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. **Install Required Packages:**
Install the dependencies by running:
```bash
pip install -r requirements.txt
```

## Usage
### Running the Project
To run the Text Emotion Classifier, follow these steps:

1. **Prepare the Dataset:**
Ensure that your `train.txt` file is in the root directory. This file should contain the text data and corresponding emotion labels, separated by a semicolon `(;)`.

2. **Run the Script:**
Execute the main script to load the data and perform emotion classification:
```bash
python main.py
```

3. **Output:**
The script will print the first few rows of the dataset to the console, showing the text samples and their associated emotion labels.

## Model Training
The model training is performed within the `main.py` script, which processes the text data, tokenizes it, and trains a Sequential model using Keras. You can modify the model architecture, training parameters, or the data processing steps within this script.
```python
# Define the model
model = Sequential()
model.add(Embedding(input_dim = len(tokenizer.word_index) + 1, output_dim=128, input_length=max_length))
model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=len(one_hot_labels[0]), activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(xtrain, ytrain, epochs=10, batch_size=32, validation_data=(xtest, ytest))
```

## Prediction
After training the model, you can use it to predict emotions from new text inputs. Implement the prediction logic in a separate script or extend `main.py` to include a prediction function.

## File Structure
Here is an overview of the project directory structure:
```lua
Text_Emotion_Classifier/
│
├── val.txt                    # This the previous version of the test data set
├── test.txt                   # Test Data set in this file for the train.txt
├── train.txt                  # The dataset file containing text and emotion labels
├── main.py                    # Main script to run the emotion classifier
├── requirements.txt           # List of dependencies
├── README.md                  # Project documentation
└── LICENSE                    # Project license
```

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

- Fork the repository & Star the repository
- Create a new branch (git checkout -b feature)
- Make your changes
- Commit your changes (git commit -am 'Add new feature')
- Push to the branch (git push origin feature)
- Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](https://github.com/Vaibhav-kesarwani/Text_Emotion_Classifier/blob/main/LICENSE) file for details.

## Acknowledgements
1. [Tensorflow](https://www.tensorflow.org/)
2. [Pandas](https://pandas.pydata.org/)
3. [Keras](https://keras.io/)
4. [Kaggle](https://www.kaggle.com/datasets)
5. [NumPy](https://numpy.org/)

## Contact
If you have any questions or suggestions, feel free to reach out to me at :
1. [GitHub](https://github.com/Vaibhav-kesarwani)
2. [Linkedin](https://www.linkedin.com/in/vaibhav-kesarwani-9b5b35252/)
3. [Twitter](https://twitter.com/Vaibhav_k__)
4. [Portfolio](https://vaibhavkesarwani.vercel.app)

**Happy Coding!** <img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Fire.png" alt="Fire" width="30" align=center /><img src="https://raw.githubusercontent.com/Tarikul-Islam-Anik/Animated-Fluent-Emojis/master/Emojis/Travel%20and%20places/Star.png" alt="Star" width="30" align=center />
