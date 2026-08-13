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


def test_manifest_uploads_last_because_it_is_the_commit_point():
    """The manifest carries the pitcher index the app routes on. Publishing it
    first (build_bundle's insertion order) meant a failure partway left the live
    app serving a new manifest against missing or stale pitcher blobs.
    """
    order = []
    def fake_upload(name, text):
        order.append(name)
    bundle = {
        "manifest.json": {"built": "x", "pitchers": [{"pitcherId": 1}]},
        "staff_board.json": {"team": "DEL_BLU", "pitchers": [{"a": 1.0}]},
        "model_artifacts.json": {"featureOrder": []},
        "pitchers/1.json": {"pitcherId": 1},
    }
    names = upload_bundle(bundle, connection_string="ignored", container="bundles",
                          upload_fn=fake_upload)
    assert order[-1] == "manifest.json"
    assert names == order
    assert set(names) == set(bundle)
