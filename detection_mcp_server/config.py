import torch
from pydantic import BaseModel, Field, model_validator


class ServerConfig(BaseModel):
    host: str = Field(default="127.0.0.1", description="Server host")
    port: int = Field(default=8000, description="Server port")
    device: str = Field(default="auto", description="Device: cuda, mps, cpu, or auto")
    use_bfloat16: bool = Field(
        default=False, description="Whether to use bfloat16 precision on supported devices"
    )

    @model_validator(mode="after")
    def resolve_device_and_bfloat16(self) -> "ServerConfig":
        if self.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        self.use_bfloat16 = self.use_bfloat16 and self.device == "cuda"
        return self


CFG = ServerConfig(
    device="auto",
    use_bfloat16=True,
)
