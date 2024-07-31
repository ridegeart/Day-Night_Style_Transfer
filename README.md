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

Refer to [Control-LoRA][3].

* Environment-Used env build from ControlNet
* Git Clone form [ComfyUI][4]
* Download `.safetensors` from [huggingface][5]
    - saved in `./models/controlnet/control-lora`
* 
* Download workflow/Revision-end with `.json` 
* Run web UI
    ```
    python main.py
    ```
* Click `Load` to load workflow
* Click `Queue_Prompt` to Start

### 

[1]: https://github.com/zyxElsa/InST
[2]: https://github.com/lllyasviel/ControlNet
[3]: https://github.com/HighCWu/ControlLoRA
[4]: https://github.com/comfyanonymous/ComfyUI.git
[5]: https://huggingface.co/stabilityai/control-lora