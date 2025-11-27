import aiohttp
from typing import Optional
from fastapi import UploadFile
import base64
import io

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
