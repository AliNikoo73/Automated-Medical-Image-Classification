import os
import csv
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2 # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # type: ignore
from tensorflow.keras.applications import (  # type: ignore
    VGG16,
    VGG19,
    Xception,
    InceptionV3,
    MobileNetV2,
    DenseNet201,
    NASNetLarge,
    InceptionResNetV2,
    ResNet152V2,
)  # Add more models as needed
from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Input,
    Dense,
    Flatten,
    BatchNormalization,
    Dropout,
    Conv2D,
    GlobalAveragePooling2D,
)  # type: ignore
from tensorflow.keras.optimizers import SGD # type: ignore
from tensorflow.keras import backend as K # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# Suppress warnings for clean output
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Print GPU information
print("\n" + "==" * 50)
print("🚀 Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
if gpus := tf.config.experimental.list_physical_devices("GPU"):
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"⚠️ {e}")
print("==" * 50 + "\n")

# Define the path to your dataset
data_path = "data/train"
test_data_path = "data/test"
classes = os.listdir(data_path)

# Specify parameters
img_size = (224, 224)  # Updated to match the expected input shape of pre-trained models
batch_size = 32
initial_epochs = 50  # Initial training with frozen base model layers
NNeuron = 256 # Number of neurons in the dense layer
DO_factor = 0.5 # Fraction of the number of input units to dropout
version = "1.0"  # Code version

# Use simple ImageDataGenerator
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
)

# Create generators for training and validation sets
train_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="training",
)

validation_generator = datagen.flow_from_directory(
    data_path,
    target_size=img_size,
    batch_size=batch_size,
    color_mode="rgb",
    class_mode="categorical",
    subset="validation",
)

# Create a generator for the test set
test_generator = datagen.flow_from_directory(
    test_data_path,
    target_size=img_size,
    batch_size=1,
    color_mode="rgb",
    class_mode="categorical",
    shuffle=False,
)

# Mapping between class indices and class labels
class_indices = test_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

# List of base models to use
base_models = {
    # "ResNet152V2": ResNet152V2,
    "DenseNet201": DenseNet201,
    # "VGG16": VGG16,
    "VGG19": VGG19,
    # "Xception": Xception,
    # "InceptionV3": InceptionV3,
    "MobileNetV2": MobileNetV2,
    # "NASNetLarge": NASNetLarge,
    # Add more models if needed
}

# Dictionary to store results for bar chart
model_metrics = {
    "Model": [],
    "Final Train Accuracy (%)": [],
    "Final Train Loss": [],
    "Final Val Accuracy (%)": [],
    "Final Val Loss": [],
    "Final Test Accuracy (%)": [],
    "Final Test Loss": [],
}


def create_fine_tune_model(base_model_name, NNeuron, DO_factor):
    """Creates a fine-tuning model based on a selected base model.

    Args:
        base_model_name (str): Name of the base model.
        NNeuron (int): Number of neurons in the dense layer.
        DO_factor (float): Dropout rate.

    Returns:
        tuple: Model, base model, and list of layer configurations.
    """
    base_model_class = base_models[base_model_name]
    base_model = base_model_class(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

    # Add new layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Global average pooling
    x = Dense(NNeuron, activation="relu")(x)  # Fully connected layer
    x = BatchNormalization()(x) # Batch normalization layer
    x = Dropout(DO_factor)(x) # Dropout layer
    predictions = Dense(train_generator.num_classes, activation="softmax")(x)  # Output layer
    
    layers_conf_list = [
        "GlobalAvgPooling",
        f"Dense_{NNeuron}_relu",
        "BatchNorm",
        f"Dropout_{DO_factor}",
    ]

    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze the base model layers initially
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model, base_model, layers_conf_list


def fine_tune_model(model, base_model, unfreeze_from):
    """Unfreezes and fine-tunes layers of the model.

    Args:
        model (Model): The compiled model.
        base_model (Model): The base model.
        unfreeze_from (int): Layer index to start unfreezing layers.

    Returns:
        Model: Fine-tuned model.
    """
    for layer in base_model.layers[:unfreeze_from]:
        layer.trainable = False
    for layer in base_model.layers[unfreeze_from:]:
        layer.trainable = True

    model.compile(optimizer=SGD(learning_rate=0.0001, momentum=0.9), loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Function to save model results
def save_results(trial_name, history, val_accuracy, val_loss, test_accuracy, test_loss):
    """Saves model results in a CSV file.

    Args:
        trial_name (str): Name of the trial.
        history (History): Training history object.
        val_accuracy (float): Validation accuracy.
        val_loss (float): Validation loss.
        test_accuracy (float): Test accuracy.
        test_loss (float): Test loss.
    """
    # Define path for the metrics CSV file
    results_csv = os.path.join(f"results/metrics/{trial_name}", f"{trial_name}_model_results.csv")
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    
    # Saving the results
    with open(results_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Epoch",
                "Train Accuracy (%)",
                "Train Loss",
                "Val Accuracy (%)",
                "Val Loss",
                "Test Accuracy (%)",
                "Test Loss",
            ]
        )
        
        # Populate CSV with epoch data
        for epoch in range(len(history.history["accuracy"])):
            train_acc = round(history.history["accuracy"][epoch] * 100, 2)
            train_loss = round(history.history["loss"][epoch], 4)
            val_acc = round(history.history["val_accuracy"][epoch] * 100, 2)
            val_loss = round(history.history["val_loss"][epoch], 4)
            
            # Set test_acc and test_loss only for the last epoch
            if epoch == len(history.history["accuracy"]) - 1:
                test_acc = round(float(test_accuracy) * 100, 2) if test_accuracy is not None else None
                test_loss = round(float(test_loss), 4) if test_loss is not None else None
            else:
                test_acc, test_loss = None, None

            # Use None to represent missing values in CSV (e.g., for earlier epochs)
            writer.writerow([
                epoch + 1,
                train_acc,
                train_loss,
                val_acc,
                val_loss,
                test_acc if test_acc is not None else "",
                test_loss if test_loss is not None else "",
            ])


# Function to plot and save accuracy and loss
def plot_metrics(history, trial_name):
    """Plots accuracy and loss.

    Args:
        history (History): Training history object.
        trial_name (str): Name of the trial.
    """
    epochs = range(1, len(history.history["accuracy"]) + 1)  # Generate epoch numbers

    # Convert accuracy to percentage
    train_accuracy = [round(acc * 100, 2) for acc in history.history["accuracy"]]
    val_accuracy = [round(acc * 100, 2) for acc in history.history["val_accuracy"]]

    # Define folder path
    plot_folder = f"results/images/{trial_name}"
    os.makedirs(plot_folder, exist_ok=True)

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, train_accuracy, label="Train Accuracy")
    plt.plot(epochs, val_accuracy, label="Validation Accuracy")
    plt.title(f"{trial_name} Model Accuracy")
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(plot_folder, f"{trial_name}_accuracy.png"))
    plt.close()

    # Plot loss
    plt.figure()
    plt.plot(epochs, history.history["loss"], label="Train Loss")
    plt.plot(epochs, history.history["val_loss"], label="Validation Loss")
    plt.title(f"{trial_name} Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(loc="upper left")
    plt.savefig(os.path.join(plot_folder, f"{trial_name}_loss.png"))
    plt.close()


def plot_predictions(trial_name, model, base_path, test_generator, index_to_class):
    """Plots predictions for random test images.

    Args:
        trial_name (str): Name of the trial.
        model (Model): Trained model.
        base_path (str): Path to save the plot.
        test_generator (DataGenerator): Test data generator.
        index_to_class (dict): Mapping from index to class label.
    """
    random_indices = random.sample(range(len(test_generator.filenames)), 20)
    selected_images, selected_labels = [], []

    for idx in random_indices:
        img, label = test_generator[idx]
        selected_images.append(img[0])
        selected_labels.append(label[0])

    predictions = model.predict(np.array(selected_images))

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(selected_images[i].squeeze(), cmap="gray")
        true_label = index_to_class[np.argmax(selected_labels[i])]
        predicted_label = index_to_class[np.argmax(predictions[i])]
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(base_path, f"{trial_name}_random_predictions.png"))
    plt.close()


def get_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """Generates a Grad-CAM heatmap for an input image.

    Args:
        model (Model): Trained model.
        img_array (array): Input image array.
        last_conv_layer_name (str): Name of the last convolutional layer.
        pred_index (int, optional): Index of the predicted class.

    Returns:
        numpy array: Grad-CAM heatmap.
    """
    grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(last_conv_layer_name).output, model.output])
    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def overlay_gradcam_heatmap(img, heatmap, alpha=0.4):
    """Overlays Grad-CAM heatmap on an image.

    Args:
        img (array): Original image.
        heatmap (array): Grad-CAM heatmap.
        alpha (float): Opacity of the heatmap.

    Returns:
        array: Image with Grad-CAM overlay.
    """
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)


# Function to display and save Grad-CAM for a single image
def display_gradcam(img_path, model, last_conv_layer_name, true_label, alpha=0.4):
    """Displays and saves Grad-CAM for a single image.

    Args:
        img_path (str): Path to the image.
        model (Model): Trained model.
        last_conv_layer_name (str): Name of the last convolutional layer.
        true_label (str): True label of the image.
        alpha (float, optional): Opacity of the heatmap. Defaults to 0.4.
    """
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    # Generate the Grad-CAM heatmap
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
    
    # Overlay the heatmap on the original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    superimposed_img = overlay_gradcam_heatmap(img, heatmap, alpha=alpha)
    
    # Define the Grad-CAM folder and save path
    gradcam_folder = "results/gradcam"
    os.makedirs(gradcam_folder, exist_ok=True)
    comparison_image_filename = os.path.join(gradcam_folder, f"gradcam_{os.path.basename(img_path)}")
    
    # Save Grad-CAM image
    cv2.imwrite(comparison_image_filename, superimposed_img)
    print(f"✅ Grad-CAM image saved to: {comparison_image_filename}")


def gradcam_for_test_images(model, test_generator, last_conv_layer_name, base_path, num_images=20):
    """Generates Grad-CAMs for a random selection of test images.

    Args:
        model (Model): Trained model.
        test_generator (DataGenerator): Test data generator.
        last_conv_layer_name (str): Name of the last conv layer.
        base_path (str): Path to save Grad-CAM images.
        num_images (int): Number of images to generate Grad-CAMs for.
    """
    class_indices = test_generator.class_indices
    index_to_class = {v: k for k, v in class_indices.items()}
    selected_images, selected_labels = [], []

    for class_name in random.sample(list(class_indices.keys()), 5):
        class_idx = class_indices[class_name]
        class_images = [i for i in range(len(test_generator.filenames)) if test_generator.classes[i] == class_idx]
        random_class_images = random.sample(class_images, 4)
        
        for img_idx in random_class_images:
            selected_images.append(test_generator.filenames[img_idx])
            selected_labels.append(class_name)

    fig, axes = plt.subplots(4, 5, figsize=(15, 12))
    axes = axes.flatten()

    for i, (img_path_relative, true_label) in enumerate(zip(selected_images, selected_labels)):
        img_path = os.path.join(test_generator.directory, img_path_relative)
        
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalize image
        
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)
        
        prediction = model.predict(img_array)
        predicted_label = index_to_class[np.argmax(prediction)]

        img = cv2.imread(img_path)
        img = cv2.resize(img, (224, 224))
        superimposed_img = overlay_gradcam_heatmap(img, heatmap, alpha=0.4)

        ax = axes[i]
        ax.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"True: {true_label}\nPred: {predicted_label}")
        ax.axis('off')

    plt.tight_layout()
    # Create the directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    comparison_image_filename = os.path.join(base_path, f"Trial_{base_model_name}_gradcam_N{NNeuron}_DO{int(DO_factor*100)}_V{version}_20_images.png")
    plt.savefig(comparison_image_filename)
    plt.close()


def find_last_conv_layer(model):
    """Finds the last Conv2D layer in the model.

    Args:
        model (Model): The trained model.

    Returns:
        str: Name of the last Conv2D layer, or None if not found.
    """
    last_conv_layer_name = next(
        (
            layer.name
            for layer in reversed(model.layers)
            if isinstance(layer, tf.keras.layers.Conv2D)
        ),
        None,
    )
    return last_conv_layer_name

# Initialize an empty list to store base model names
base_model_names_used = []

# Perform trials with each base model
for base_model_name in base_models.keys():
    trial_name = f"Trial_{base_model_name}_N{NNeuron}_DO{int(DO_factor*100)}_EPOCHS{initial_epochs}_V{version}"
    print("\n" + "==" * 50)
    print(f"🚀 Starting {trial_name} with base model: {base_model_name}")
    print("==" * 50 + "\n")
    
    # Append base model name to list if not already added
    if base_model_name not in base_model_names_used:
        base_model_names_used.append(base_model_name)

    # Step 1: Train the model with frozen base model layers
    model, base_model, layers_conf_list = create_fine_tune_model(base_model_name, NNeuron, DO_factor)

    # Retrieve the last convolutional layer name for Grad-CAM
    last_conv_layer_name = find_last_conv_layer(base_model)

    # Skip Grad-CAM if no Conv2D layer is found
    if last_conv_layer_name is None:
        print(f"⚠️ Skipping Grad-CAM for {base_model_name} due to no Conv2D layer.")
        continue  

    # Train the model
    history = model.fit(train_generator, epochs=initial_epochs, validation_data=validation_generator)

    # Evaluate the model on validation data
    val_loss, val_accuracy = model.evaluate(validation_generator)

    # Evaluate the model on test data
    test_loss, test_accuracy = model.evaluate(test_generator)

    # Save metrics to the metrics folder
    save_results(trial_name, history, val_accuracy, val_loss, test_accuracy, test_loss)

    # Save accuracy and loss plots to the images folder
    plot_metrics(history, trial_name)

    # Save the fine-tuned model to the models folder
    model.save(os.path.join(f"models/{trial_name}", f"{trial_name}_model.h5"))

    # Save predictions for 20 random images in the images folder
    plot_predictions(trial_name, model, f"results/images/{trial_name}", test_generator, index_to_class)

    # Generate Grad-CAM for test images and save in the gradcam folder
    gradcam_for_test_images(model, test_generator, last_conv_layer_name, f"results/gradcam/{trial_name}", num_images=20)

    # Store final metrics for bar chart (not saved to files yet)
    model_metrics["Model"].append(base_model_name)
    model_metrics["Final Train Accuracy (%)"].append(round(history.history["accuracy"][-1] * 100, 2))
    model_metrics["Final Train Loss"].append(round(history.history["loss"][-1], 4))
    model_metrics["Final Val Accuracy (%)"].append(round(history.history["val_accuracy"][-1] * 100, 2))
    model_metrics["Final Val Loss"].append(round(history.history["val_loss"][-1], 4))
    model_metrics["Final Test Accuracy (%)"].append(round(test_accuracy * 100, 2))
    model_metrics["Final Test Loss"].append(round(test_loss, 4))

    print(f"✅ {trial_name} completed and results saved.")

    # Clear session to prevent memory buildup
    tf.keras.backend.clear_session()

print("\n" + "==" * 50)
print("🎉 All model trials completed!")
print("==" * 50 + "\n")

# Plot bar chart comparing accuracy and loss of each model
fig, ax = plt.subplots(1, 2, figsize=(18, 8))

# Colors for the bars
train_color = "lightblue"
val_color = "lightgreen"
test_color = "salmon"

# Plot accuracy bar chart
width = 0.25  # Width of each bar

r1 = np.arange(len(model_metrics["Model"]))  # X positions for train bars
r2 = [x + width for x in r1]  # X positions for validation bars
r3 = [x + 2 * width for x in r1]  # X positions for test bars

ax[0].bar(
    r1,
    model_metrics["Final Train Accuracy (%)"],
    color=train_color,
    width=width,
    label="Train Accuracy",
)
ax[0].bar(
    r2,
    model_metrics["Final Val Accuracy (%)"],
    color=val_color,
    width=width,
    label="Validation Accuracy",
)
ax[0].bar(
    r3,
    model_metrics["Final Test Accuracy (%)"],
    color=test_color,
    width=width,
    label="Test Accuracy",
)

# Display the accuracy values on top of the bars
for i in range(len(model_metrics["Model"])):
    ax[0].text(
        r1[i],
        model_metrics["Final Train Accuracy (%)"][i] + 0.5,
        f"{model_metrics['Final Train Accuracy (%)'][i]:.2f}%",
        ha="center",
        color="black",
    )
    ax[0].text(
        r2[i],
        model_metrics["Final Val Accuracy (%)"][i] + 0.5,
        f"{model_metrics['Final Val Accuracy (%)'][i]:.2f}%",
        ha="center",
        color="black",
    )
    ax[0].text(
        r3[i],
        model_metrics["Final Test Accuracy (%)"][i] + 0.5,
        f"{model_metrics['Final Test Accuracy (%)'][i]:.2f}%",
        ha="center",
        color="black",
    )

# Labeling and title for accuracy chart
ax[0].set_title("Final Accuracy of Models", fontsize=14)
ax[0].set_ylabel("Accuracy (%)")
ax[0].set_xticks([r + width for r in range(len(model_metrics["Model"]))])
ax[0].set_xticklabels(model_metrics["Model"])
# Place the legend outside the accuracy chart
ax[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[0].tick_params(axis="x", rotation=45)

# Plot loss bar chart
ax[1].bar(
    r1,
    model_metrics["Final Train Loss"],
    color=train_color,
    width=width,
    label="Train Loss",
)
ax[1].bar(
    r2,
    model_metrics["Final Val Loss"],
    color=val_color,
    width=width,
    label="Validation Loss",
)
ax[1].bar(
    r3,
    model_metrics["Final Test Loss"],
    color=test_color,
    width=width,
    label="Test Loss",
)

# Display the loss values on top of the bars
for i in range(len(model_metrics["Model"])):
    ax[1].text(
        r1[i],
        model_metrics["Final Train Loss"][i] + 0.01,
        f"{model_metrics['Final Train Loss'][i]:.4f}",
        ha="center",
        color="black",
    )
    ax[1].text(
        r2[i],
        model_metrics["Final Val Loss"][i] + 0.01,
        f"{model_metrics['Final Val Loss'][i]:.4f}",
        ha="center",
        color="black",
    )
    ax[1].text(
        r3[i],
        model_metrics["Final Test Loss"][i] + 0.01,
        f"{model_metrics['Final Test Loss'][i]:.4f}",
        ha="center",
        color="black",
    )

# Labeling and title for loss chart
ax[1].set_title("Final Loss of Models", fontsize=14)
ax[1].set_ylabel("Loss")
ax[1].set_xticks([r + width for r in range(len(model_metrics["Model"]))])
ax[1].set_xticklabels(model_metrics["Model"])
# Place the legend outside the loss chart
ax[1].legend(loc='upper left', bbox_to_anchor=(1, 1))
ax[1].tick_params(axis="x", rotation=45)

# After the loop completes, create a summary name using the list of base models
base_model_summary = "_".join(base_model_names_used)  # Concatenate model names with underscores
chart_name = f"Trial_{base_model_summary}_comparison bar chart_N{NNeuron}_DO{int(DO_factor*100)}_EPOCHS{initial_epochs}_V{version}.png"

# Once the loop is done and the final bar chart is plotted:
plt.tight_layout()
plt.savefig(
    os.path.join(f"results/images/{trial_name}", chart_name)
)
# plt.show()
plt.close()