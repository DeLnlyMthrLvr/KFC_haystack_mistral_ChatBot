# KFC_haystack_mistral_ChatBot
## Alessandro Corvi

### Introduction
This project demonstrates a retriever augmented generation which gathers information from the file `menu.json` and all other uploaded files. The project was developed around the `Mixtral-8x7B-Instruct-v0.1` model with the aid of the pipeline `haystack`and uses `streamlit` to generate a user interface.

### Running
#### VOX_mistral_haystack.ipynb
Running `VOX_mistral_haystack.ipynb` can be done by using google colab (with at least a free tier T4 gpu). The program will ask for an HuggingFace access token which can be retrieved from the account settings on their website.

#### app.py
Google Colab is not advised to be used in order to run the StreamLit application and could cause problems. I utilised a pod from runpods to run on my computer, giving me a virtual machine with enough power to run the tasks. but, because this pod is paid hourly, I am unable to keep it running continuously; when the process is running, anyone can visit the webapp from anywhere by using a URL.
The port `8501` needs to be open for StreamLit to function. 

In order to run the application only the notebook `Run.ipynb` shall be compiled.