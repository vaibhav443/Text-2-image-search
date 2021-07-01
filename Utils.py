import torch
import numpy as np
import pandas as pd
from PIL import Image
from sentence_transformers import util
from simple_tokenizer import *
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from CLIP.clip import clip

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=True)
input_resolution = model.input_resolution.item()
context_length = model.context_length.item()
vocab_size = model.vocab_size.item()
preprocess = Compose([
    Resize(input_resolution, interpolation=Image.BICUBIC),
    CenterCrop(input_resolution),
    ToTensor()
])


def encode_search_query(search_query):
    tokenizer = SimpleTokenizer()
    text_tokens = [tokenizer.encode(desc) for desc in [search_query]]
    text_input = torch.zeros(len(text_tokens), model.context_length, dtype=torch.long)
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']

    for i, tokens in enumerate(text_tokens):
        tokens = [sot_token] + tokens + [eot_token]
        text_input[i, :len(tokens)] = torch.tensor(tokens)

    text_input = text_input.cuda()
    with torch.no_grad():
        text_features = model.encode_text(text_input).float()

    text_features /= text_features.norm(dim=-1, keepdim=True)

    return text_features.cpu().numpy()


def find_best_matches(text_features, photo_features, image_attributes, results_count=3):
    # Compute the similarity between the search query and each photo using the Cosine similarity
    # similarities = (photo_features @ text_features.T).squeeze(1)
    similarities = util.pytorch_cos_sim(photo_features, text_features)
    similarity = np.array([score[0] for score in similarities.numpy()])
    # Sort the photos by their similarity score
    best_photo_idx = (-similarity).argsort()

    # Return the photo IDs of the best matches
    return [image_attributes[i] for i in best_photo_idx[:results_count]], np.sort(similarity)[::-1][:results_count]


def get_image_features(directory):
    image_tuples = []
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    for filename in [filename for filename in os.listdir(directory) if
                     filename.endswith(".png") or filename.endswith(".jpg")]:
        image = preprocess(Image.open(os.path.join(directory, filename)).convert("RGB"))
        image_tuples.append((filename, image))
    images = [element[1] for element in image_tuples]
    image_attributes = [element[0] for element in image_tuples]
    image_input = torch.tensor(np.stack(images)).cuda()
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    with torch.no_grad():
        image_features = model.encode_image(image_input).float()

    return image_features, image_attributes
