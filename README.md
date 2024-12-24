# Texture Recognition Bot

This repository contains a Telegram bot (TBot_resnet50.py) that identifies textures from user-uploaded images. It utilizes a ResNet50 model fine-tuned on the [Describable Textures Dataset (DTD)](https://www.robots.ox.ac.uk/~vgg/data/dtd/). You can run it locally or explore the model directly in Jupyter/Colab.

## Features

* **Texture Classification**: Send an image, and the bot returns the most likely texture class (e.g., “striped,” “dotted,” “scaly,” etc.) with a confidence score.
    
* **Pretrained Neural Network**: Fine-tuned ResNet50 model specialized in recognizing various texture patterns.
    
* **Easy Deployment**: Install requirements, insert your Telegram token, and run!

## Setup and Installation

**1. Clone the repository:**
 
       git clone https://github.com/YourUsername/your-texture-bot.git

**2. Insert your Telegram token**
* Open TBot_resnet50.py and find the line where the bot’s token should be placed.

* Obtain the token from [BotFather](https://t.me/BotFather) (creating a new bot, following their instructions).

**3. Install dependencies:**

      pip install -r requirements.txt

**4. Run the bot:**

      python TBot_resnet50.py

Make sure you have Python installed and you’ve downloaded this repository locally. Also, if you already have obsolete numpy and other old libraries on your computer, it can cause to some conflicts with other libraries from requirements. So, it can require to reinstall some of them in order to get the last vertion for every lib.

## Using the Bot

1. Open your Telegram app and navigate to your bot’s chat.
2. Type /start (if the bot doesn’t greet you automatically).
3. Tap the “Screen” button the bot displays.
4. Follow the instructions to upload an image with a texture.
5. Receive a reply indicating the predicted texture class and its probability.

## Example Results

Include any screenshots or brief examples of the bot’s predictions here.

![IMG_8620](https://github.com/user-attachments/assets/fd4fb4a7-7433-4f5b-8a31-9f5bb3edf90f)

![IMG_8623](https://github.com/user-attachments/assets/49ac953b-1f22-41df-914d-f14386a27c4a)

![IMG_8625](https://github.com/user-attachments/assets/426bc550-c5f9-477d-9059-3a8ab7f2b673)

![IMG_8628](https://github.com/user-attachments/assets/19b3447b-5295-40bf-85b5-2ab8e500c43b)

![IMG_8630](https://github.com/user-attachments/assets/e25d5168-e266-41f3-bba8-aa5513d2fb77)


## Live Demo

A live version of this bot is currently running on my VPS server. You can try it out by messaging [@screen_texture_bot](https://t.me/screen_texture_bot) in Telegram.


## Model Details

* resnet50_model.ipynb/py:
Contains the model code. You can run it in Google Colab, Jupyter Lab, or other environments to see how it works without the Telegram interface.

* model_training.ipynb/py:
Shows how the model was trained from scratch or fine-tuned. You can train it yourself locally or in Google Colab (runs for 5 epochs by default).


# Algorithm Description

**1. Imports and Setup**

**Import necessary libraries:**

* **Torch & Torchvision** for model loading, preprocessing, and inference.
* **PIL** for image handling.
* **python-telegram-bot** library for creating and managing Telegram bot commands and messages.
* **nest_asyncio** to allow asynchronous code (if running in environments like Jupyter).

**2. Global Variables**

**device** is determined (CPU or GPU) for running model inference.
**classes** is a list of all texture classes recognized by this model (47 in total).
**num_classes** is set to the number of classes in **classes**.

**3. Model Loading**

* A pre-trained ResNet50 model is retrieved from **torchvision.models** with **pretrained=True**.
* The final classification layer (model.fc) is replaced with a new linear layer for 47 classes.
* The script attempts to load the trained weights from **resnet50_dtd_split1.pth**. If the file is found, it loads them into the model; otherwise, it proceeds without loading those weights.
* The model is moved to the appropriate device (CPU or GPU) and set to evaluation mode.

**4. Image Preprocessing**

A transforms pipeline is defined:
* **Resize** the image to 224×224 pixels.
* **Convert** it to a PyTorch tensor.
* **Normalize** it with the standard ImageNet mean and std.

**5. Telegram Bot Handlers**

a. start_command Handler
* Responds to the /start command.
* Greets the user and provides an inline button labeled “screen”.

b. button_callback Handler
* Listens for clicks on the inline buttons.
* If the user taps “screen”, it sends instructions to upload a photo.

c. photo_handler Handler
* Triggered when the user sends a photo.
* Takes the largest version of the photo ([-1] from the list).
* Downloads the photo locally (temp_image.jpg).
* Loads it with PIL, runs it through the preprocessing pipeline, and performs inference with the loaded ResNet50 model.
* Uses softmax to get probabilities, picks the top class via argmax, and calculates the confidence.
* Sends back a message with the predicted class and confidence score.
* Offers the “screen” button again so the user can classify another image.

**6. Main Bot Launch**
* Checks if __name__ == "__main__", meaning the script is being run directly (not imported).
* Retrieves or sets TELEGRAM_TOKEN.
* Builds the application (ApplicationBuilder().token(TELEGRAM_TOKEN).build()).
* Registers all the handlers for commands, callbacks, and photos:
    * /start -> start_command
    * Inline button -> button_callback
    * Photo messages -> photo_handler
* Finally, calls application.run_polling() to start long polling, so the bot remains active, listening for user messages and events.

   
