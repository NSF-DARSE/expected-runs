"""Serialize bundle files and upload them to Blob storage.

`upload_fn` is injectable so tests never need real Azure credentials or
network access. When not supplied, a real Azure Blob uploader is built from
`connection_string` / `container`.
"""
import json
from webapp_publisher.build_bundle import to_native


def _default_upload_fn(connection_string, container):
    from azure.storage.blob import BlobServiceClient, ContentSettings
    svc = BlobServiceClient.from_connection_string(connection_string)
    cont = svc.get_container_client(container)

    def upload(name, text):
        cont.upload_blob(
            name=name, data=text.encode("utf-8"), overwrite=True,
            content_settings=ContentSettings(content_type="application/json", cache_control="no-cache"),
        )
    return upload


MANIFEST = "manifest.json"


def upload_bundle(bundle, *, connection_string, container, upload_fn=None):
    """Upload every bundle file, with manifest.json LAST.

    The manifest is the commit point: it carries the pitcher index the app routes
    on, so publishing it before the blobs it references means a failure partway
    leaves the live app pointed at files that are missing or stale. Uploading it
    last keeps the last good data live until the new data is fully in place.
    """
    upload_fn = upload_fn or _default_upload_fn(connection_string, container)
    names = [n for n in bundle if n != MANIFEST] + [n for n in bundle if n == MANIFEST]
    uploaded = []
    for name in names:
        text = json.dumps(to_native(bundle[name]), allow_nan=False, separators=(",", ":"))
        upload_fn(name, text)
        uploaded.append(name)
    return uploaded
