import aiohttp
from typing import Optional
from fastapi import UploadFile
import base64
import io
import requests
import cv2
import numpy as np
import os
import uuid
from starlette.datastructures import UploadFile as StarletteUploadFile
from App.Helpers.GeminiEmbedding import getDescriptionByAi


class CFWorkerService:
    def __init__(self, worker_url: str):
        self.worker_url = worker_url

    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = "",
        width: int = 512,
        height: int = 512,
        num_steps: int = 20,
        guidance: float = 10,
        action: str = "generate",
        image_file: Optional[UploadFile] = None,
        init_file: Optional[UploadFile] = None,
        strength: Optional[float] = 0.5,
        seed: Optional[int] = None,
    ) -> str:

        data = aiohttp.FormData()

        # TEXT FIELDS -> phải thêm như thế này
        fields = {
            "action": action,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": str(width),
            "height": str(height),
            "num_steps": str(num_steps),
            "guidance": str(guidance),
        }

        if seed is not None:
            fields["seed"] = str(seed)
        if strength is not None:
            fields["strength"] = str(strength)

        for key, val in fields.items():
            data.add_field(key, val)

        # FILE FIELDS -> THIS IS THE FIX
        async def add_uploadfile(field_name, upload: UploadFile):
            if not upload:
                return

            file_bytes = await upload.read()
            file_like = io.BytesIO(file_bytes)

            data.add_field(
                field_name,
                file_like,
                filename=upload.filename,
                content_type=upload.content_type,
            )

        await add_uploadfile("image", image_file)
        await add_uploadfile("init_image", init_file)

        async with aiohttp.ClientSession() as session:
            async with session.post(self.worker_url, data=data) as resp:
                if resp.status != 200:
                    msg = await resp.text()
                    raise RuntimeError(
                        f"CF Worker AI error {resp.status}: {msg}"
                    )
                content = await resp.read()
                return base64.b64encode(content).decode("utf-8")

    def merge_frames_grid(self, frames_base64, grid_shape=(2, 5), frame_size=(512, 512), padding=2):
        images = []
        for b64 in frames_base64:
            img_data = base64.b64decode(b64.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            h, w = img.shape[:2]
            scale = min(frame_size[0] / h, frame_size[1] / w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            top = (frame_size[0] - new_h) // 2
            bottom = frame_size[0] - new_h - top
            left = (frame_size[1] - new_w) // 2
            right = frame_size[1] - new_w - left
            padded = cv2.copyMakeBorder(
                resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
            )

            images.append(padded)

        rows, cols = grid_shape
        grid_rows = []
        for r in range(rows):
            row_imgs = images[r * cols:(r + 1) * cols]
            row_imgs_padded = [
                cv2.copyMakeBorder(img, 0, 0, 0, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                for img in row_imgs
            ]
            grid_rows.append(np.hstack(row_imgs_padded))

        grid_image = np.vstack(grid_rows)

        _, buffer = cv2.imencode(".jpg", grid_image, [int(cv2.IMWRITE_JPEG_QUALITY), 98])

        # return base64
        # return base64.b64encode(buffer).decode("utf-8")

        file_like = io.BytesIO(buffer.tobytes())

        # Trả về UploadFile
        return StarletteUploadFile(filename="test.jpg", file=file_like)

        # return base64.b64encode(buffer).decode("utf-8")

    async def extract_and_describe(self, video_url: str):
        TEMP_PATH = f"temp_video_{uuid.uuid4().hex}.mp4"

        try:
            resp = requests.get(video_url)
            if resp.status_code != 200:
                return {"error": "Failed to download video"}

            with open(TEMP_PATH, "wb") as f:
                f.write(resp.content)

            cap = cv2.VideoCapture(TEMP_PATH)
            if not cap.isOpened():
                return {"error": "Cannot open video"}

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration_sec = total_frames / fps if fps > 0 else 0

            n_frames = 10 if duration_sec < 60 else 20 if duration_sec <= 180 else 30
            frame_indices = sorted(set(int(i * total_frames / n_frames) for i in range(n_frames)))

            frames_base64 = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                _, buffer = cv2.imencode(".jpg", frame)
                frames_base64.append(
                    "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")
                )
            cap.release()

            grid_cols = min(5, len(frames_base64))
            grid_rows = (len(frames_base64) + grid_cols - 1) // grid_cols

            grid_base64 = self.merge_frames_grid(frames_base64, (grid_rows, grid_cols))
            system_prompt = """
            You are a video captioning assistant.
            
            ## TASK
            Given an image that is a collage of multiple frames from a video, generate a coherent English description of the video content. 
            Treat the image as a representation of the video flow, and write the description as if you are watching the video in real time.
            
            ## RULES
            - Describe visible objects, people, actions, movements, and settings.
            - Do NOT mention that the input is an image, collage, or frames.
            - Do NOT invent emotions, background stories, or assumptions.
            - Keep it concise, clear, and flowing like a short narration.
            - Use complete sentences and natural language.
            """

            user_prompt = """
            Describe the video content naturally and coherently, as if you are watching it in real time. 
Focus only on visible actions, objects, people, movements, and settings. 
Do NOT mention frames, images, or that it is a collage. 
Do NOT invent emotions, background stories, or assumptions. 
Keep the description concise, flowing like a short narration in complete sentences.
"""
            description = await getDescriptionByAi(grid_base64,system_prompt, user_prompt)
            return description


            # call by cloudfalre worker ai
            # payload = {
            #     "image": grid_base64,
            #     "prompt": (
            #         "Describe the video naturally and coherently, as if watching it in real time. "
            #         "Do NOT mention frames or images — write it like a real flowing video."
            #     ),
            #     "top_p": 0.8,
            #     "top_k": 20,
            #     "presence_penalty": 1.0,
            #     "max_tokens": 1000,
            # }
            #
            # worker_resp = requests.post(self.worker_url, json=payload)
            #
            # if worker_resp.status_code != 200:
            #     return {"error": f"Worker error: {worker_resp.text}"}
            #
            # # Lấy JSON
            # data = worker_resp.json()
            #
            # # Only description
            # description = data.get("description", None)
            # return description

        except Exception as e:
            return {"error": str(e)}

        finally:
            if os.path.exists(TEMP_PATH):
                try:
                    os.remove(TEMP_PATH)
                except:
                    pass