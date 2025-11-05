# GenAI Tools: CIFAR-10 Image Classification with a Vision-Language Model (gemma3:4b)

This repo samples 100 CIFAR-10 images (10 per class), sends each image to an OpenAI‑compatible endpoint at **https://ai.sooners.us** using the **gemma3:4b** model, and logs predictions to compute **accuracy** and a **confusion matrix**. 


## 1. Setup

### Requirements
- Python 3.9+ recommended
- `pip install -r requirements.txt`

### Create your SoonerAI account & API key
1. Visit **https://ai.sooners.us** and sign up with your OU email.
2. After logging in: **Settings → Account → API Keys**.
3. Create a new API key and copy it.

### Create local env file (DO NOT commit)
Create `~/.soonerai.env` and put the following code into the file.
SOONERAI_API_KEY=your_key_here
SOONERAI_BASE_URL=https://ai.sooners.us
SOONERAI_MODEL=gemma3:4b

# run the chatbot
python cifar10_classify.py


## Analysis

Setup summary:

10 images per CIFAR-10.

Model: gemma3:4b via https://ai.sooners.us/api/chat/completions.

Temperature = 0.0 for deterministic output.


# Failure modes:

Small objects or heavy background clutter.
Low-contrast animal photos.

# Conclusion
Explicit, short definitions yield the highest usable accuracy while maintaining valid single-word outputs.
