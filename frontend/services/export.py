import json
from models.upload import Upload

def uploads_to_json(uploads: list[Upload]):
    return json.dumps([u.to_dict() for u in uploads], indent=2)