from flask import Flask, render_template, url_for, Response
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.utils import save_image
import os
import io
from PIL import Image


image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
app = Flask(__name__)


def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


os.makedirs("static", exist_ok=True)
statsdir = "static"


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = "generated-images-{0:0=4d}.png".format(index)
    save_image(denorm(fake_images), os.path.join(statsdir, fake_fname), nrow=8)
    print("Saving", fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))
        plt.show()


latent_size = 128

device = torch.device("cpu")

generator = nn.Sequential(
    nn.ConvTranspose2d(
        latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False
    ),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh()
    # out: 3 x 64 x 64
)


generator.load_state_dict(torch.load("G.ckpt", map_location="cpu"))


@app.route("/", methods=["GET"])
def hello():

    fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
    save_samples(0, fixed_latent, show=False)

    img = Image.open("static/generated-images-0000.png")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    return Response(response=byte_im, mimetype="image/png")


if __name__ == "__main__":
    port = int(os.environ('PORT',5000)))
    app.run(debug=True, port=5000,host='0.0.0.0')

