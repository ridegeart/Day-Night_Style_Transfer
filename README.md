# Day-Night_Style_Transfer

Employ style transfer techniques to transform an image into different lighting scenarios (night, twilight), utilizing Google Earth's solar irradiation data at a particular time as the style guide
/#style transfer #Stable Diffusion Model #ControlNet

## Stable Diffusion Model

The following experiments several Stable Diffusion Models.

### InST

Refer to [InST][1].  

* Build environment
* Train Data：Style 1 images
* Train

    ```
    python main.py --base configs/stable-diffusion/v1-finetune.yaml
                -t 
                --actual_resume ./models/sd/sd-v1-4.ckpt
                -n ldm 
                --gpus 0, 
                --data_root /path/to/directory/with/images
    ```
* Generate-`InST.ipynb`
    - `model.embedding_manager.load`：path of trained weight (in log/embeddings.pt).
    - `content_dir`/`style_dir`：path of content/style, single images. 

### ControlNet

Refer to [ControlNet][2].

* Build environment
* Run web UI or other method `gradio_****2image.py`
    ```
    python gradio_hed2image.py
    ```
* Prompt
    ```
    An aerial view of a parking space road at night with no street lights,a late night scene,nighttime atmosphere,nothing that shines,no light source
    ```
* Adavanced setting
    - resolution：384
    - images：4

### Control-LoRA

Refer to [Control-LoRA][3]. [Tutorial][6]

* Environment-Used env build from ControlNet
* Git Clone form [ComfyUI][4]
* Download `.safetensors` from [huggingface][5]
    - save in `./models/controlnet/control-lora`
* Download `sd_xl_base_1.0.safetensors` from [huggingface][7]
    - save in `./models/checkpoint`
* Download workflow/Revision-end with `.json` 
* Install custom nodes [Tutorial][8]
* Run web UI
    ```
    python main.py
    ```
* Click `Load` to load workflow
* Upadate UnInstall node
* Click `Queue_Prompt` to Start

## Comparative analysis

Perform a comparative analysis between the style-transferred image (downscaled) and the original image patch. Segment the entire runway and apply brightness adjustments to individual segments.

### Change.py

1. Central Ratio
```python
def calculate_local_ratio(V_A, V_B, block_size=9):
```

2. Global Ratio
```python
def calculate_each_ratio(V_A, V_B):
```
* Input V channel of each image
* img_A：style_transfer image
* img_B：Original image

[1]: https://github.com/zyxElsa/InST
[2]: https://github.com/lllyasviel/ControlNet
[3]: https://github.com/HighCWu/ControlLoRA
[4]: https://github.com/comfyanonymous/ComfyUI.git
[5]: https://huggingface.co/stabilityai/control-lora
[6]: https://youtu.be/uK51kvxFkhc?si=-XrWl89Z_Yedszjd
[7]: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/tree/main
[8]: https://ivonblog.com/posts/comfyui-install-extensions/