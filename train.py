# Gradient Deployments: Fashion-MNIST Example
#
# This is part of our basic deployments example (https://github.com/gradient-ai/Deployments) that shows
#
# 1: Create and train a TensorFlow deep learning model using Workflows
# 2: Deploy the model using Deployments
# 3: Send inference data to the model and receive correct output
#
# This script is part of step 1 and is called from the Workflow. It does the model training.
# It is based on content from our Workflows tutorial at https://github.com/gradient-ai/fashionmnist
#
# Last updated: Dec 07th 2021

# Input parameters
# import os
# # MODEL_DIR = os.path.abspath(os.environ.get('MODEL_DIR', os.getcwd() + 'models'))
# MODEL_DIR = os.path.abspath(os.environ.get('MODEL_DIR', os.getcwd() + '/models'))
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")
tokenizer = AutoTokenizer.from_pretrained("ramsrigouthamg/t5-large-paraphraser-diverse-high-quality")

# import torch
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device ", device)
# model = model.to(device)

# # Save model
# export_path = os.path.join(MODEL_DIR)
# print('export_path = {}\n'.format(export_path))
#
# #see the current directory
# import os
#
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
#
# # filename = f'{export_path}/model_pkl'
# # pickle.dump(model, open(filename, 'wb'))
import pickle
#
# # create an iterator object with write permission - model.pkl
# with open(f'{export_path}/model_pkl.pkl', 'wb') as files:
#     pickle.dump(model, files)

#these two works
with open('/trained-model/1/models/model_pkl.pkl', 'wb') as files:
    pickle.dump(model, files)
# with open('model_pkl', 'wb') as files:  #this works
#     pickle.dump(model, files)
# cwd = os.getcwd()  # Get the current working directory (cwd)
# files = os.listdir(cwd)  # Get all the files in that directory
# print("Files in %r: %s" % (cwd, files))
# # print('\nModel saved to ' + MODEL_DIR)
