from webapp_publisher.upload import upload_bundle

def test_upload_bundle_serializes_each_file_and_reports_names():
    calls = {}
    def fake_upload(name, text):
        calls[name] = text
    bundle = {"manifest.json": {"built": "x"}, "staff_board.json": {"team": "DEL_BLU", "pitchers": [{"a": float('nan')}]}}
    names = upload_bundle(bundle, connection_string="ignored", container="bundles", upload_fn=fake_upload)
    assert set(names) == {"manifest.json", "staff_board.json"}
    # NaN must serialize to null (JSON-valid)
    assert "NaN" not in calls["staff_board.json"]
    assert "null" in calls["staff_board.json"]
