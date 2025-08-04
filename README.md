# <p align="center">` Codename-RVC-Fork 🍇 4 ` </p>
## <p align="center">Based on Applio</p>

<p align="center"> ㅤㅤ👇 You can join my discord server below ( RVC / AI Audio friendly ) 👇ㅤㅤ </p>

</p>
<p align="center">
  <a href="https://discord.gg/SkwcMjxwMq" target="_blank"> Codename's Sanctuary</a>
</p>



## A lil bit more about the project:

### This fork is pretty much my personal take on Applio. ✨
``You could say.. A more advanced features-rich Applio ~ With my lil twist.``
<br/>
``But If you have any ideas, want to pr or collaborate, feel free to do so!``
<br/>
ㅤ
<br/>
# ⚠️ㅤ**Datasets must be processed properly!** ㅤ⚠️
- Normalization
- Peak-compression
- Silence-truncation
- 'simple' method chosen for preprocessing ( 3sec segments. )
- Ideally, you want to use 'post' as norm option in the ui.
Expect issues with PESQ and data alignment If the following requirements are not met.
<br/>``THESE REQUIREMENTS ARE ESPECIALLY IMPORTANT FOR RINGFORMER ARCHITECTURE.``
ㅤ
<br/>

# **Fork's features:**
 
- Hold-Out type validation mechanism during training. ( L1 MEL, mrSTFT, PESQ, SI-SDR )  ` In between epochs. `
 
- BF16-AMP, TF32, FP32 Training modes available.  ` BF16 & TF32 require Ampere or newer GPUs. `<br/>
`BF16 and TF32 can be used simultaneously for extra speed gains.`
> NOTE: BF16 is used by default ( and bf16-AdamW ). If unsupported hardware detected, switched back to FP32. Inference is only in FP32.

 
- Support for 'Spin' embedder.  ` Needs proper pretrains. `
 
- Ability to choose an optimizer.  ` ( Currently supporting: AdamW, AdamW_BF16, RAdam, Ranger21, DiffGrad ) `
 
- Double-update strategy for Discriminator.
 
- Support for custom input-samples used during training for live-preview of model's reconstruction performance.
 
- Mel spectrogram %-based similarity metric.
 
- Support for Multi-scale and classic L1 mel loss.
 
- Support for the following vocoders: HiFi-GAN, MRF-HiFi-gan, Refine-GAN, RingFormer.  ` ( And their respective pretrains. ) `
 
- Checkpointing and various speed / memory optimizations compared to og RVC.
 
- New logging mechanism for losses: Average loss per epoch logged as the standard loss, <br/>and rolling average loss over 50 steps to evaluate general trends and the model's performance over time.
 
- From-ui quick tweaks; lr for g/d, schedulers, linear warmup etc.

 
 
<br/>``⚠️ 1: HiFi-gan is the stock rvc/applio vocoder, hence it's what you use for og pretrains and hifigan-based customs. ``
<br/>``⚠️ 2: MRF-HiFi-GAN, Refine-GAN and RingFormer require new pretrained models. They can't be used with original rvc's G/D pretrains. ``
 <br/>
 
 
✨ to-do list ✨
> - None
 
💡 Ideas / concepts 💡
> - Currently none. Open to your ideas ~
 
 
### ❗ For contact, please join my discord server ❗
 
 
## Getting Started:

### 1. Installation of the Fork

Run the installation script:

- Double-click `run-install.bat`.

### 2. Running the Fork

Start Applio using:

- Double-click `run-fork.bat`.
 
This launches the Gradio interface in your default browser.

### 3. Optional: TensorBoard Monitoring

To monitor training or visualize data:
- Run the " run_tensorboard_in_model_folder.bat " file from logs folder and paste in there path to your model's folder </br>( containing 'eval' folder or tfevents file/s. )</br>If it doesn't work for you due to blocked port, open up CMD with admin rights and use this command:</br>`` netsh advfirewall firewall add rule name="Open Port 25565" dir=in action=allow protocol=TCP localport=25565 ``


## Disclaimer
``The creators of the original Applio repository, Applio's contributors, and the maintainer of this fork (Codename;0), built upon Applio, are not responsible for any legal issues, damages, or consequences arising from the use of this repository or the content generated from it. By using this fork, you acknowledge that:``

- The use of this fork is at your own risk.
- This repository is intended solely for educational, and experimental purposes.
- Any misuse, including but not limited to illegal activities or violation of third-party rights, <br/> is not the responsibility of the original creators, contributors, or this fork’s maintainer.
- You willingly agree to comply with this repository's [Terms of Use](https://github.com/codename0og/codename-rvc-fork-3/blob/main/TERMS_OF_USE.md)
