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

## Live Demo

A live version of this bot is currently running on my VPS server. You can try it out by messaging [@screen_texture_bot](https://t.me/screen_texture_bot) in Telegram.


## Model Details

* resnet50_model.ipynb/py:
Contains the model code. You can run it in Google Colab, Jupyter Lab, or other environments to see how it works without the Telegram interface.

* model_training.ipynb/py:
Shows how the model was trained from scratch or fine-tuned. You can train it yourself locally or in Google Colab (runs for 5 epochs by default).
   
