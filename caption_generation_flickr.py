import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Concatenate, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from nltk.translate.bleu_score import sentence_bleu
import nltk
import pickle


# ------------------- Step 1: Load Dataset -------------------
def load_dataset_hf(data_dir):
    captions_file = os.path.join(data_dir, "captions.txt")
    images_dir = os.path.join(data_dir, "Images")

    # Check if the files exist
    if not os.path.exists(captions_file):
        raise FileNotFoundError(f"Captions file not found at: {captions_file}")
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found at: {images_dir}")

    # Read captions
    captions = pd.read_csv(captions_file, delimiter=',')
    captions.columns = ["image", "caption"]  # Rename columns for consistency
    return captions, images_dir


# ------------------- Step 2: Preprocess Captions -------------------
def preprocess_captions(captions):
    # Add start and end tokens
    captions['caption'] = captions['caption'].apply(lambda x: f'<start> {x.strip()} <end>')

    # Initialize tokenizer with OOV token
    tokenizer = Tokenizer(num_words=10000, oov_token='<unk>')
    tokenizer.fit_on_texts(captions['caption'])  # Fit on processed captions

    # Manually ensure <start> and <end> tokens are in the word index
    if '<start>' not in tokenizer.word_index:
        tokenizer.word_index['<start>'] = len(tokenizer.word_index) + 1
    if '<end>' not in tokenizer.word_index:
        tokenizer.word_index['<end>'] = len(tokenizer.word_index) + 1

    # Debug: Print the tokenizer's vocabulary
    #print("Final Tokenizer Vocabulary:", tokenizer.word_index)

    # Convert captions to numeric sequences
    sequences = tokenizer.texts_to_sequences(captions['caption'])
    max_length = max(len(seq) for seq in sequences)
    captions_seq = pad_sequences(sequences, maxlen=max_length, padding='post')

    # Validate the presence of <start> and <end> tokens
    if '<start>' not in tokenizer.word_index or '<end>' not in tokenizer.word_index:
        raise ValueError("Special tokens <start> or <end> are missing from the tokenizer vocabulary.")

    return tokenizer, captions_seq, max_length



# ------------------- Step 3: Extract Image Features -------------------
def extract_features(image_path, model):
    img = Image.open(image_path).resize((299, 299))
    img = np.array(img).astype('float32')
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    features = model.predict(img)
    return features.squeeze()

def create_feature_extractor():
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    return model

def extract_and_save_features(images_dir, feature_extractor, feature_file):
    if os.path.exists(feature_file):
        print(f"Features already saved at {feature_file}. Skipping extraction.")
        with open(feature_file, 'rb') as f:
            features = pickle.load(f)
        return features

    print("Extracting image features...")
    features = {}
    for img_id in os.listdir(images_dir):
        img_path = os.path.join(images_dir, img_id)
        if img_id.lower().endswith(('.jpg', '.jpeg', '.png')):
            features[img_id] = extract_features(img_path, feature_extractor)

    with open(feature_file, 'wb') as f:
        pickle.dump(features, f)
    print(f"Features saved to {feature_file}.")
    return features


# ------------------- Step 4: Create the Model -------------------
def create_caption_model(vocab_size, embedding_dim, units, max_length):
    image_input = Input(shape=(2048,), name="image_features_input")  # Image features
    caption_input = Input(shape=(max_length,), name="caption_input")  # Partial captions

    # Image features processing
    image_dense = Dense(units, activation='relu')(image_input)

    # Caption processing
    caption_embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, mask_zero=True)(caption_input)
    caption_lstm = LSTM(units)(caption_embedding)

    # Combine image and caption features
    combined = Concatenate()([image_dense, caption_lstm])
    output = Dense(vocab_size, activation='softmax')(combined)

    model = Model(inputs=[image_input, caption_input], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


# ------------------- Step 5: Data Generator -------------------
def data_generator(captions, features, tokenizer, max_length, batch_size, vocab_size):
    while True:
        for i in range(0, len(captions), batch_size):
            X1, X2, y = [], [], []
            for j in range(i, min(i + batch_size, len(captions))):
                image_id = captions.iloc[j, 0]
                feature = features[image_id]
                seq = tokenizer.texts_to_sequences([captions.iloc[j, 1]])[0]

                for k in range(1, len(seq)):
                    X1.append(feature)  # Image features
                    X2.append(seq[:k])  # Partial caption sequence
                    y.append(to_categorical(seq[k], num_classes=vocab_size))  # Next word
            yield ([np.array(X1), pad_sequences(X2, maxlen=max_length)], np.array(y))


# ------------------- Step 6: Generate Captions -------------------
def generate_caption(model, feature, tokenizer, max_length):
    # Start with the <start> token
    caption = [tokenizer.word_index['<start>']]
    for _ in range(max_length):
        sequence = pad_sequences([caption], maxlen=max_length, padding='post')
        yhat = model.predict([np.expand_dims(feature, axis=0), sequence], verbose=0)
        word_idx = np.argmax(yhat)
        word = tokenizer.index_word.get(word_idx, '<unk>')  # Use <unk> if the word index is missing
        caption.append(word_idx)
        if word == '<end>':
            break
    return ' '.join([tokenizer.index_word.get(idx, '<unk>') for idx in caption if idx > 0])


# ------------------- Step 7: Main Function -------------------
def main():
    nltk.download('punkt')
    
    # Specify the data directory
    data_dir = "./data"
    captions, images_dir = load_dataset_hf(data_dir)
    
    # Preprocess captions
    tokenizer, captions_seq, max_length = preprocess_captions(captions)
    vocab_size = len(tokenizer.word_index) + 1
    embedding_dim = 256
    units = 512

    # Save tokenizer for future use
    tokenizer_file = os.path.join(data_dir, "tokenizer.pkl")
    if not os.path.exists(tokenizer_file):
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
    
    # Extract features
    feature_file = os.path.join(data_dir, "image_features.pkl")
    feature_extractor = create_feature_extractor()
    features = extract_and_save_features(images_dir, feature_extractor, feature_file)
    
    # Load or train model
    model_file = os.path.join(data_dir, "caption_model.h5")
    if os.path.exists(model_file):
        print("Loading saved model...")
        caption_model = load_model(model_file)
    else:
        print("Training model...")
        caption_model = create_caption_model(vocab_size, embedding_dim, units, max_length)
        batch_size = 64
        steps_per_epoch = len(captions) // batch_size

        caption_model.fit(
            data_generator(captions, features, tokenizer, max_length, batch_size, vocab_size),
            steps_per_epoch=steps_per_epoch,
            epochs=10
        )
        caption_model.save(model_file)
        print(f"Model saved at {model_file}.")

    # Test caption generation
    test_image = os.path.join(images_dir, captions.iloc[0, 0])
    test_feature = features[captions.iloc[0, 0]]
    generated_caption = generate_caption(caption_model, test_feature, tokenizer, max_length)

    print(f"Generated Caption: {generated_caption}")
    img = Image.open(test_image)
    plt.imshow(img)
    plt.title(generated_caption)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
