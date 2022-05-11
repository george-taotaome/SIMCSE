# -*- coding: utf-8 -*-

from typing import Optional
from pydantic import BaseModel

class Queue(BaseModel):
    text: str
    limit: Optional[int] = 1
