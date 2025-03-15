import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

import nest_asyncio  
nest_asyncio.apply()

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    CallbackQueryHandler,
    MessageHandler,
    filters,
)

import os

# ---------------------
# LOGGING
# ---------------------
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ---------------------
# GLOBAL VARIABLES
# ---------------------

# Device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Alphabetical list of 47 texture classes (DTD)
classes = [
    "banded", "blotchy", "braided", "bubbly", "bumpy", 
    "chequered", "cobwebbed", "cracked", "crosshatched", "crystalline",
    "dotted", "fibrous", "flecked", "freckled", "frilly", 
    "gauzy", "grid", "grooved", "honeycombed", "interlaced",
    "knitted", "lacelike", "lined", "marbled", "matted", 
    "meshed", "paisley", "perforated", "pitted", "pleated",
    "polka-dotted", "porous", "potholed", "scaly", "smeared",
    "spiralled", "sprinkled", "stained", "stratified", "striped",
    "studded", "swirly", "veined", "waffled", "woven",
    "wrinkled", "zigzagged"
]
num_classes = len(classes)  

# ---------------------
# MODEL LOADING
# ---------------------
model = models.resnet50(pretrained=True)

# Change the last layer for 47 classes (DTD)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Load trained weights (the file 'resnet50_dtd_split1.pth' must be located nearby)
weights_path = "resnet50_dtd_split1.pth"
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location=device))
    logger.info("Model weights loaded successfully.")
else:
    logger.warning(f"Could not find the weights file {weights_path}! Continuing without loading...")

model.to(device)
model.eval()

# ---------------------
# IMAGE PREPROCESSING
# ---------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])

# ---------------------
# HANDLERS
# ---------------------

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for the /start command.
    Greets the user and offers the "screen" button.
    """
    keyboard = [
        [InlineKeyboardButton("screen", callback_data="screen")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        text=(
            "Hello! I’m a bot for texture recognition.\n\n"
            "Press the button below to send me an image, "
            "and I’ll tell you which texture it is."
        ),
        reply_markup=reply_markup
    )

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for inline button presses (in particular, the "screen" button).
    """
    query = update.callback_query
    await query.answer()

    if query.data == "screen":
        text_instructions = (
            "Please send me a photo (as a separate message), and I’ll tell you "
            "which texture it is.\n\n"
            "Once you get the result, you can press 'screen' again "
            "if you want to scan another image."
        )
        await query.edit_message_text(text=text_instructions)

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for photos (images) sent by the user.
    """
    # Get the photo object (Telegram sends photos in several sizes).
    photo = update.message.photo[-1]  

    # Download the image to a temporary file
    image_path = "temp_image.jpg"
    photo_file = await photo.get_file()
    await photo_file.download_to_drive(image_path)

    # Load with PIL
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        predicted_class = classes[pred_idx]
        confidence = probs[0, pred_idx].item()

    result_text = (
        f"Predicted class: {predicted_class}\n"
        f"Confidence: {confidence:.2f}"
    )
    await update.message.reply_text(result_text)

    # Offer the "screen" button again
    keyboard = [
        [InlineKeyboardButton("screen", callback_data="screen")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        text="If you want to recognize another image, press 'screen'.",
        reply_markup=reply_markup
    )

# ---------------------
# MAIN (bot launch)
# ---------------------
if __name__ == "__main__":
    # # Connect to Telegram
    # TELEGRAM_TOKEN = "YOUR_TOKEN"

    # Get Telegram token from environment variable
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN_TEXTURE")

    if TELEGRAM_TOKEN is None:
        raise ValueError("TELEGRAM_TOKEN_TEXTURE is not set. Please provide it as an environment variable.")

    application = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))

    # Run the bot (run_polling is asynchronous internally, but we call it synchronously)
    application.run_polling()
