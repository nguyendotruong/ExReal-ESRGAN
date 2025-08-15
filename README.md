# ExReal-ESRGAN

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/nguyendotruong/ExReal-ESRGAN.git
   cd ExReal-ESRGAN
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   python setup.py develop
   ```

---

## 🚀 Inference

```bash
python inference_realesrgan.py -n ExReal-ESRGAN -i images --outscale 4 --tile 512
```