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
* 

[1]: https://github.com/zyxElsa/InST
[2]: https://github.com/lllyasviel/ControlNet?tab=readme-ov-file