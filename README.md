To use, add to beginning of notebook:
"""
%%capture
!rm -rf toolbox
!git clone https://github.com/zentralwerkstatt/toolbox
!pip3 install git+https://github.com/openai/CLIP.git
!pip3 install umap-learn filetype
"""