################################################################################
# Based on AI Jason's YouTube tutorial
# URL: https://youtu.be/_j7JEDWuqLE?si=0vQ_yitVcOu7wXgY
#
# Create a `.env` file with "OPENAI_API_KEY=<OPENAI_PRIVATE_KEY>"

# To run the code:
# streamlit run app.py
################################################################################
import os

import requests
import soundfile
import streamlit as st
import torch
from dotenv import find_dotenv, load_dotenv
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
from fairseq.utils import move_to_cuda
from langchain import LLMChain, OpenAI, PromptTemplate
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

# Load Token
load_dotenv(find_dotenv())

MAX_TOKENS = os.getenv('MAX_TOKENS', 50)


def image2txt(input_img):
    """Generate text with a given image's URL."""

    # download image
    # image = Image.open(requests.get(input_img, stream=True).raw).convert('RGB')
    image = Image.open(input_img).convert('RGB')

    # unconditional image captioning
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    inputs = processor(image, return_tensors="pt").to("cuda", torch.float16)

    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", torch_dtype=torch.float16).to("cuda")
    out = model.generate(**inputs, max_new_tokens=20)
    # tensor to text
    text = processor.decode(out[0], skip_special_tokens=True)
    return text


def text2story(text, max_tokens=MAX_TOKENS):
    """Using a LLM to generate a story."""

    template = """
    You are a story teller. You can generate a short story based on a simple narrative, the story should be no more than {num_words} words.
    CONTEXT: {scenario}
    STORY:
    """
    promote = PromptTemplate(template=template, input_variables=['scenario', 'num_words'])

    story_llm = LLMChain(
        llm=OpenAI(model='text-curie-001', temperature=1, max_tokens=MAX_TOKENS),
        prompt=promote,
        verbose=True,
    )
    story = story_llm.predict(scenario=text, num_words=MAX_TOKENS)
    return story


def text2speech(text, path_output='story.flac'):
    """Text to speech, save as a flac audio file."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        "facebook/fastspeech2-en-ljspeech",
        arg_overrides={"vocoder": "hifigan", "fp16": False}
    )
    model = models[0].to(device)
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator([model], cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    sample = move_to_cuda(sample) if torch.cuda.is_available() else sample
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)

    # Save audio file
    soundfile.write(path_output, wav.cpu(), rate, format='flac', subtype='PCM_24')


def main():
    """Create a main page for users to upload image"""
    if not os.path.exists('outputs'):
        os.mkdir('outputs')

    st.set_page_config(page_title='Image 2 Audio')
    st.header('From an image to an audio story')
    uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'png'])

    if uploaded_file is not None:
        print(uploaded_file)

        # Save and display image
        bytes_data = uploaded_file.getvalue()
        path_img = os.path.join('outputs', uploaded_file.name)
        with open(path_img, 'wb') as file:
            file.write(bytes_data)
        st.image(uploaded_file, caption='Uploaded image', use_column_width=True)

        # Generate and save story
        text = image2txt(path_img)
        story = text2story(text)
        path_story = os.path.splitext(path_img)[0] + '.txt'
        with open(path_story, 'w') as file:
            file.write(story)

        with st.expander('Scenario'):
            st.write(text)
        with st.expander('Story'):
            st.write(story)

        # Generate and save audio
        path_audio = os.path.splitext(path_img)[0] + '.flac'
        text2speech(story, path_audio)
        st.audio(path_audio)


if __name__ == '__main__':
    main()
