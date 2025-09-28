# Generating figures for the essay and saving them as PNGs.
# This code will:
# - Create Figure 1: pipeline diagram (blocks + arrows)
# - Create Figure 2: Fourier illustration of downsampling (spectra plots)
# - Create Figure 3: Synthetic case study images: HR face, LR probe, bicubic SR output
# - Create Figure 4: Synthetic ROC curves for baseline, SR, identity-aware SR
#
# All plots use matplotlib (no seaborn) and each chart is its own figure.
# Files will be saved to /mnt/data and links will be provided after execution.

from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
import os
import os

folder = "face_recognition_figures"

# Create the folder if it does not exist
if not os.path.exists(folder):
    os.makedirs(folder)

out_dir = "/mnt/data/face_recognition_figures"
os.makedirs(out_dir, exist_ok=True)

# Figure 1: Pipeline diagram
fig, ax = plt.subplots(figsize=(10,3.5))
ax.set_xlim(0,10)
ax.set_ylim(0,3.5)
ax.axis('off')

# Boxes coordinates: (x,y,width,height)
boxes = [
    (0.2,1.6,1.6,0.8,"Scene\n& Optics"),
    (2.0,1.6,1.6,0.8,"Sensor\nSampling"),
    (3.8,1.6,1.8,0.8,"Preprocessing\n(denoise, SR)"),
    (5.8,1.6,1.6,0.8,"Detection\n& Alignment"),
    (7.6,1.6,1.6,0.8,"Embedding\nExtraction"),
    (9.4,1.6,0.8,0.8,"Matching\nDecision")
]

for x,y,w,h,label in boxes:
    rect = plt.Rectangle((x,y), w, h, fill=False)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center')

# Arrows
arrow_props = dict(arrowstyle="->", linewidth=1)
ax.annotate("", xy=(1.9,2.0), xytext=(1.8,2.0), arrowprops=arrow_props)
ax.annotate("", xy=(3.6,2.0), xytext=(3.5,2.0), arrowprops=arrow_props)
ax.annotate("", xy=(5.6,2.0), xytext=(5.5,2.0), arrowprops=arrow_props)
ax.annotate("", xy=(7.4,2.0), xytext=(7.3,2.0), arrowprops=arrow_props)
ax.annotate("", xy=(9.0,2.0), xytext=(8.9,2.0), arrowprops=arrow_props)

ax.set_title("Figure 1 — End-to-end face recognition pipeline (high-level)")

fig1_path = os.path.join(out_dir, "figure1_pipeline.png")
fig.savefig(fig1_path, bbox_inches='tight', dpi=200)
plt.close(fig)

# Figure 2: Fourier illustration of downsampling
# Create radial frequency spectra representation (synthetic)
size = 256
u = np.linspace(-1,1,size)
U,V = np.meshgrid(u,u)
R = np.sqrt(U**2 + V**2)
# Original spectrum (bandlimited)
orig_spec = np.exp(-(R*10)**2)
# After low-pass filtering (stronger attenuation)
lp_spec = np.exp(-(R*6)**2)
# Aliased spectrum (simulate replicas)
replica = np.roll(orig_spec, 40, axis=0) + np.roll(orig_spec, -30, axis=1)

fig, axes = plt.subplots(1,3, figsize=(12,4))
axes[0].imshow(orig_spec, origin='lower')
axes[0].set_title("Original Spectrum")
axes[0].axis('off')
axes[1].imshow(lp_spec, origin='lower')
axes[1].set_title("Low-pass Filtered")
axes[1].axis('off')
axes[2].imshow(lp_spec + 0.4*replica, origin='lower')
axes[2].set_title("After Undersampling (aliasing)")
axes[2].axis('off')

fig.suptitle("Figure 2 — Spectral effects of downsampling and aliasing", y=0.95)
fig2_path = os.path.join(out_dir, "figure2_spectra.png")
fig.savefig(fig2_path, bbox_inches='tight', dpi=200)
plt.close(fig)

# Figure 3: Synthetic face images (HR, LR, SR)
# Create a synthetic "face" as an image
def create_simple_face(size=256):
    img = Image.new("L", (size,size), color=200)  # light gray background
    draw = ImageDraw.Draw(img)
    # face ellipse
    draw.ellipse((size*0.2, size*0.15, size*0.8, size*0.85), fill=240, outline=0)
    # eyes
    eye_w = size*0.06
    eye_h = size*0.06
    draw.ellipse((size*0.35-eye_w, size*0.4-eye_h, size*0.35+eye_w, size*0.4+eye_h), fill=30)
    draw.ellipse((size*0.65-eye_w, size*0.4-eye_h, size*0.65+eye_w, size*0.4+eye_h), fill=30)
    # mouth
    draw.arc((size*0.35, size*0.6, size*0.65, size*0.75), start=0, end=180, fill=60, width=4)
    # nose
    draw.polygon([(size*0.5, size*0.45), (size*0.47, size*0.55), (size*0.53, size*0.55)], fill=80)
    # small texture: add noise speckles to simulate skin texture
    arr = np.array(img).astype(np.uint8)
    noise = (np.random.randn(*arr.shape) * 2).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

hr = create_simple_face(256)
# Create LR probe by resizing down to 32x32 and then upsample back to 256 for visualization
lr = hr.resize((32,32), resample=Image.BICUBIC)
sr_bicubic = lr.resize((256,256), resample=Image.BICUBIC)

# Compose side-by-side
w,h = hr.size
compound = Image.new("L", (w*3+40, h))
compound.paste(hr, (0,0))
compound.paste(lr.resize((w,w)), (w+20,0))
compound.paste(sr_bicubic, (2*w+40,0))
# Add labels using PIL
draw = ImageDraw.Draw(compound)
draw.text((20,h-20), "HR (gallery)", fill=0)
draw.text((w+40,h-20), "LR probe (32x32)", fill=0)
draw.text((2*w+60,h-20), "SR (bicubic)", fill=0)

fig3_path = os.path.join(out_dir, "figure3_case_study.png")
compound.save(fig3_path)

# Figure 4: Synthetic ROC curves
fpr = np.linspace(0,1,200)
# Baseline weaker
tpr_baseline = 1 - np.exp(-5*(0.3 + (1-fpr)**1.5))
# SR slightly improved
tpr_sr = 1 - np.exp(-5*(0.4 + (1-fpr)**1.6))
# Identity-aware SR best
tpr_id = 1 - np.exp(-6*(0.55 + (1-fpr)**1.7))

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(fpr, tpr_baseline, linewidth=1.5, label="Baseline VIS model on LR probes")
ax.plot(fpr, tpr_sr, linewidth=1.5, label="SR pre-processing + VIS model")
ax.plot(fpr, tpr_id, linewidth=1.5, label="Identity-aware SR + fine-tune")
ax.plot([0,1],[0,1], linestyle='--', linewidth=0.8)
ax.set_xlabel("False Positive Rate (FPR)")
ax.set_ylabel("True Positive Rate (TPR)")
ax.set_title("Figure 4 — Synthetic ROC curves (illustrative)")
ax.legend(loc="lower right")
fig4_path = os.path.join(out_dir, "figure4_roc.png")
fig.savefig(fig4_path, bbox_inches='tight', dpi=200)
plt.close(fig)

# List files created
created_files = [fig1_path, fig2_path, fig3_path, fig4_path]
created_files

print("Script has finished running. Check face_recognition_figures folder!")
from PIL import Image
import os

folder = "face_recognition_figures"
for file in os.listdir(folder):
    if file.endswith(".png"):
        img = Image.open(os.path.join(folder, file))
        img.show()  # Opens each figure in your default image viewer
