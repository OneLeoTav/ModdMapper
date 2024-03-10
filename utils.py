import streamlit as st
import cv2
from numpy import asarray
import os

from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import pandas as pd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, ColorBar
from bokeh.transform import LinearColorMapper
from bokeh.palettes import Viridis256
from bokeh.embed import components

# Handles buttons behavior
def start_webcam_feed():
    st.session_state['run'] = True

def stop_webcam_feed():
	st.session_state['run'] = False


# The CNN Architecture to be imported for inference
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=5),
            nn.BatchNorm2d(8),
            nn.ReLU(),           
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(),   
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )

        self.linear_layer1 = nn.Sequential(
            nn.Linear(16 * 9 * 9, 1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )

        self.linear_layer2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(0.4),
            nn.ReLU()
        )
        
        self.linear_layer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(0.3),
            nn.ReLU()
        )
        
        
        self.linear_layer4 = nn.Linear(256, 7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # x = self.conv3(x)
        # x = self.conv4(x)
        x = x.view(-1, 16 * 9 * 9)
        x = self.linear_layer1(x)
        x = self.linear_layer2(x)
        x = self.linear_layer3(x)
        x = self.linear_layer4(x)
        return x
    
label_mapping = {
    "angry": 0,
    "disgusted": 1,
    "fearful": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprised": 6
}

# Our trained model 
@st.cache_resource
def load_model(
    model_path: str,  # Path to the saved model file (string)
    device: Union[str, torch.device] = 'cpu',  # Device to load the model onto (string or torch.device object)
) -> torch.nn.Module:  # Returns a PyTorch model (torch.nn.Module)
    """
    Load a pre-trained CNN model from the specified file path onto the specified device.

    Args:
        model_path (str): Path to the saved model file.
        device (Union[str, torch.device], optional): Device to load the model onto. Defaults to 'cpu'.

    Returns:
        torch.nn.Module: Loaded CNN model.
    """
    model = CNN().to(device)  # Instantiate the CNN model and move it to the specified device
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the model's state dict from the specified file
    model.eval()  # Set the model to evaluation mode
    return model.to(device)  # Move the model to the specified device and return it


def dataframe_to_dict(dictionary: dict) -> pd.DataFrame:
    """
    Convert a dictionary containing emotion counts to a DataFrame with emotion names and their percentages.

    Parameters:
    dictionary (dict): A dictionary containing emotion counts, where keys are emotion names and values are counts.

    Returns:
    pd.DataFrame: A DataFrame with two columns - 'Emotion' and 'Percentage'. 
                  'Emotion' column contains the emotion names, and 'Percentage' column contains 
                  the percentage of each emotion with respect to the total count, rounded to one decimal place 
                  and followed by the percentage symbol '%'.
    """
    emotion_emoji = {
    'surprised': '😮',
    'angry': '😡',
    'neutral': '😐',
    'happy': '😊',
    'sad': '😢',
    'fearful': '😨',
    'disgusted': '🤢'
    }
    # Create a DataFrame from the dictionary 'dictionary' with columns 'Emotion' and 'Percentage'
    df = pd.DataFrame(list(dictionary.items()), columns=['Emotion', 'Value'])
    
    # Calculate the total sum of values
    total_sum = df['Value'].sum()
    
    # Calculate the percentage of each emotion with respect to the total
    df['Percentage'] = (df['Value'] / total_sum) * 100
    
    # Sorting here facilitates the plotting phase
    df = df.sort_values(by=['Percentage'], ascending=True, ignore_index=True)
    
    # For a more convenient dusplay: round the percentage values to one decimal place and add the percentage symbol '%'
    df['Percentage_%'] = df['Percentage'].apply(lambda x: '{:.1f}%'.format(round(x, 1)))


    df['Emotion'] = [f"{emotion} {emotion_emoji[emotion]}" for emotion in df['Emotion']]

    
    return df

def create_bokeh_plot(df: pd.DataFrame) -> figure:
    """
    Create a Bokeh horizontal bar chart showing the emotion distribution.

    Parameters:
    df (pd.DataFrame): A DataFrame containing emotion names and their percentages.

    Returns:
    bokeh.plotting.figure: A Bokeh figure showing the emotion distribution.
    """
    TOOLTIPS = [("Detected emotion", "@Emotion"),
                ("Percentage over the session", "@Percentage{0.0}%"),
               ]

    # Bokeh figure
    p = figure(
        y_range=df['Emotion'],
        # plot_width=800,
        # plot_height=400,
        title="Your Emotion Report below 👇",
        toolbar_location=None,
        tools='tap',
        background_fill_alpha=0.2,
        tooltips=TOOLTIPS
    )

    mapper = LinearColorMapper(palette=Viridis256, low=0, high=100)
    color_bar = ColorBar(color_mapper=mapper, location=(0,0))

    p.hbar(
        y='Emotion',
        right='Percentage',
        height=0.5,
        source=ColumnDataSource(df),
        line_color='white',
        color={'field': 'Percentage', 'transform': mapper}
    )
    
    p.text(
        y='Emotion',
        x='Percentage',
        text='Percentage_%',
        text_font_size='10pt',
        text_baseline='middle',
        text_align='left',
        source=ColumnDataSource(df),
        x_offset=5
    )

    p.toolbar.logo = None
    p.toolbar_location = None
    p.grid.grid_line_alpha = 0.0
    p.axis.minor_tick_out = 0

    # Axis abels
    p.border_fill_color = (255, 255, 255, 0)
    p.xaxis.axis_line_width = 0
    p.xaxis.visible = False
    p.xaxis.axis_line_color = None
    # div, js = components(p)
    # return div, js
    return p