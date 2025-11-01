import uvicorn

from app.api.endpoints import app
from app.core.config import settings


if __name__ == '__main__':
    uvicorn.run(
        app,
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=False
    )
