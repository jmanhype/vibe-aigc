from PIL import Image

img = Image.open("generated_outputs/backend_video.webp")
print(f"Frames: {img.n_frames}")

frames = []
for i in range(img.n_frames):
    img.seek(i)
    frames.append(img.copy())

frames[0].save(
    "generated_outputs/backend_video.gif",
    save_all=True,
    append_images=frames[1:],
    duration=62,
    loop=0
)
print("Saved as backend_video.gif!")
