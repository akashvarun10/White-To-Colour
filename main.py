
# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.responses import StreamingResponse
# from colorizers import siggraph17, preprocess_img, postprocess_tens
# from PIL import Image
# import numpy as np
# import torch
# import io

# app = FastAPI()

# colorizer = siggraph17(pretrained=True).eval()


# def colorize_image(file: UploadFile, use_gpu: bool = False):
#     try:
#         contents = file.file.read()
#         img = Image.open(io.BytesIO(contents)).convert("RGB")
#         img_np = np.array(img)

#         (tens_l_orig, tens_l_rs) = preprocess_img(img_np, HW=(256, 256))

#         if use_gpu:
#             tens_l_rs = tens_l_rs.cuda()

#         with torch.no_grad():
#             out_ab = colorizer(tens_l_rs).cpu()

#         out_img = postprocess_tens(tens_l_orig, out_ab)
#         out_img_pil = Image.fromarray((out_img * 255).astype(np.uint8))

#         # Save the result image to bytes
#         img_bytes = io.BytesIO()
#         out_img_pil.save(img_bytes, format="PNG")

#         return StreamingResponse(io.BytesIO(img_bytes.getvalue()), media_type="image/png")

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/colorize")
# async def colorize_endpoint(file: UploadFile = File(...), use_gpu: bool = False):
#     return colorize_image(file, use_gpu)

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from colorizers import siggraph17, preprocess_img, postprocess_tens
from PIL import Image
import numpy as np
import torch
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Enable CORS (adjust origins as needed for security in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

colorizer = siggraph17(pretrained=True).eval()

def colorize_image(file: UploadFile, use_gpu: bool = False):
    try:
        contents = file.file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(img)

        (tens_l_orig, tens_l_rs) = preprocess_img(img_np, HW=(256, 256))

        if use_gpu:
            tens_l_rs = tens_l_rs.cuda()

        with torch.no_grad():
            out_ab = colorizer(tens_l_rs).cpu()

        out_img = postprocess_tens(tens_l_orig, out_ab)
        out_img_pil = Image.fromarray((out_img * 255).astype(np.uint8))

        # Save the result image to bytes
        img_bytes = io.BytesIO()
        out_img_pil.save(img_bytes, format="PNG")

        return StreamingResponse(io.BytesIO(img_bytes.getvalue()), media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/colorize")
async def colorize_endpoint(file: UploadFile = File(...), use_gpu: bool = False):
    return colorize_image(file, use_gpu)
