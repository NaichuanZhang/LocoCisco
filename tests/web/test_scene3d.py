"""Static checks for the 3D house (homedemo/web/static/3d + vendor + assets). No browser is needed:
these guard the offline guarantee (every import-map target, vendored import, model and texture the 3D
code loads exists on disk and nothing imports or fetches an absolute http(s) URL) and the asset budget."""
from __future__ import annotations

import json
import re
import shutil
import struct
import subprocess
from pathlib import Path

import httpx
import pytest

from homedemo.web import server as web
from homedemo.web.dev_fake import FakeRuntime

STATIC = Path(web.__file__).resolve().parent / "static"
INDEX = STATIC / "index.html"
VENDOR = STATIC / "vendor" / "three"
SCENE_JS = sorted((STATIC / "3d").glob("*.js"))
SIZE_BUDGET = 15 * 1024 * 1024


def import_map() -> dict[str, str]:
    m = re.search(r'<script type="importmap">\s*(\{.*?\})\s*</script>', INDEX.read_text(encoding="utf-8"), re.S)
    assert m, "index.html has no import map"
    return json.loads(m.group(1))["imports"]


def resolve_import(spec: str, from_file: Path) -> Path:
    """Resolve a module specifier the way the browser will: import map first, then relative to the file."""
    imports = import_map()
    if spec in imports:
        return (STATIC.parent / imports[spec].lstrip("./")).resolve()
    for prefix, target in imports.items():
        if prefix.endswith("/") and spec.startswith(prefix):
            return (STATIC.parent / target.lstrip("./") / spec[len(prefix):]).resolve()
    assert spec.startswith(("./", "../")), f"{from_file.name}: bare specifier {spec!r} is not in the import map"
    return (from_file.parent / spec).resolve()


def js_imports(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    return re.findall(r"""^\s*(?:import|export)\b[^'";]*?\bfrom\s+['"]([^'"]+)['"]""", text, re.M)


def glb_json(path: Path) -> dict:
    data = path.read_bytes()
    magic, _version, _length = struct.unpack_from("<III", data, 0)
    assert magic == 0x46546C67, f"{path.name} is not a GLB"
    chunk_len, chunk_type = struct.unpack_from("<II", data, 12)
    assert chunk_type == 0x4E4F534A
    return json.loads(data[20:20 + chunk_len])


# --------------------------------------------------------------------------- import map + vendored files
def test_import_map_targets_exist():
    imports = import_map()
    assert set(imports) == {"three", "three/addons/"}
    for spec, target in imports.items():
        assert target.startswith("./static/"), f"{spec} -> {target} must be a local relative URL"
        p = STATIC.parent / target.lstrip("./")
        assert p.exists(), f"import map {spec} -> {target} is missing on disk"
        assert p.is_dir() if spec.endswith("/") else p.is_file()


def test_vendored_three_layout_and_version():
    for rel in ("LICENSE", "build/three.module.js", "build/three.core.js", "examples/jsm/controls/OrbitControls.js",
                "examples/jsm/loaders/GLTFLoader.js", "examples/jsm/renderers/CSS2DRenderer.js"):
        assert (VENDOR / rel).is_file(), rel
    assert "MIT" in (VENDOR / "LICENSE").read_text(encoding="utf-8")
    assert json.loads((VENDOR / "package.json").read_text(encoding="utf-8"))["version"] == "0.186.1"
    assert re.search(r"REVISION\s*=\s*'186'", (VENDOR / "build" / "three.core.js").read_text(encoding="utf-8"))


def test_every_vendored_import_resolves():
    files = [*VENDOR.rglob("*.js")]
    assert files
    for f in files:
        for spec in js_imports(f):
            target = resolve_import(spec, f)
            assert target.is_file(), f"{f.relative_to(STATIC)} imports {spec!r} -> {target} (missing)"


def test_scene_module_imports_resolve():
    assert SCENE_JS, "no 3D modules found"
    for f in SCENE_JS:
        for spec in js_imports(f):
            target = resolve_import(spec, f)
            assert target.is_file(), f"{f.name} imports {spec!r} -> missing {target}"


def test_no_static_js_or_html_imports_or_fetches_remote_urls():
    """Vendor comments may mention https URLs; actual imports, fetches and script/link tags must not."""
    bad = [
        re.compile(r"""^\s*import\b[^'";]*['"]https?://""", re.M),
        re.compile(r"""import\(\s*['"]https?://"""),
        re.compile(r"""fetch\(\s*['"]https?://"""),
        re.compile(r"""new\s+EventSource\(\s*['"]https?://"""),
        re.compile(r"""<(script|link|img)\b[^>]*\b(src|href)=["']https?://""", re.I),
        re.compile(r"""url\(\s*['"]?https?://"""),
    ]
    files = [f for f in STATIC.rglob("*") if f.is_file() and f.suffix in {".js", ".html", ".css", ".gltf"}]
    assert len(files) > 10
    for f in files:
        text = f.read_text(encoding="utf-8")
        for rx in bad:
            assert not rx.search(text), f"{f.relative_to(STATIC)} loads a remote URL: {rx.pattern}"


# --------------------------------------------------------------------------- assets the 3D code loads
def referenced_asset_paths() -> set[Path]:
    """Every static/assets/... path a 3D module names, plus model names joined to their directories."""
    out: set[Path] = set()
    for f in SCENE_JS:
        text = f.read_text(encoding="utf-8")
        for rel in re.findall(r"static/assets/[\w./-]+\.(?:glb|gltf|png|bin)", text):
            out.add(STATIC.parent / rel)
    furniture = (STATIC / "3d" / "furniture.js").read_text(encoding="utf-8")
    models = re.search(r"FURNITURE_MODELS\s*=\s*\[(.*?)\];", furniture, re.S)
    assert models
    for name in re.findall(r"'([\w]+)'", models.group(1)):
        out.add(STATIC / "assets" / "furniture" / f"{name}.gltf")
    people = (STATIC / "3d" / "people.js").read_text(encoding="utf-8")
    for name in re.findall(r"'(character-[\w-]+\.glb)'", people):
        out.add(STATIC / "assets" / "people" / name)
    for name in re.findall(r"CAT_MODEL\s*=\s*'([\w-]+\.glb)'", people):
        out.add(STATIC / "assets" / "pets" / name)
    return out


def test_every_model_and_texture_the_3d_code_loads_exists():
    paths = referenced_asset_paths()
    assert any(p.suffix == ".glb" for p in paths) and any(p.suffix == ".gltf" for p in paths)
    for p in sorted(paths):
        assert p.is_file(), f"3D code loads {p.relative_to(STATIC)} but it is missing"
        if p.suffix == ".gltf":
            doc = json.loads(p.read_text(encoding="utf-8"))
            for uri in [b["uri"] for b in doc.get("buffers", []) if "uri" in b] + [i["uri"] for i in doc.get("images", []) if "uri" in i]:
                assert not uri.startswith(("http:", "https:", "data:")), f"{p.name}: {uri}"
                assert (p.parent / uri).is_file(), f"{p.name} references missing {uri}"
        elif p.suffix == ".glb":
            doc = glb_json(p)
            for img in doc.get("images", []):
                if "uri" in img:
                    assert (p.parent / img["uri"]).is_file(), f"{p.name} references missing texture {img['uri']}"
            for buf in doc.get("buffers", []):
                assert "uri" not in buf, f"{p.name}: external buffer {buf['uri']}"


def test_furniture_layout_uses_only_copied_models():
    text = (STATIC / "3d" / "furniture.js").read_text(encoding="utf-8")
    models = set(re.findall(r"'([\w]+)'", re.search(r"FURNITURE_MODELS\s*=\s*\[(.*?)\];", text, re.S).group(1)))
    used = set(re.findall(r"model:\s*'([\w]+)'", text))
    assert used <= models, f"layout uses models not in FURNITURE_MODELS: {used - models}"


def test_character_models_have_the_animations_the_code_plays():
    needed = {"idle", "walk", "sit", "interact-right", "emote-yes"}
    for glb in sorted((STATIC / "assets" / "people").glob("character-*.glb")):
        names = {a["name"] for a in glb_json(glb).get("animations", [])}
        assert needed <= names, f"{glb.name} lacks {needed - names}"
    cat = glb_json(STATIC / "assets" / "pets" / "animal-cat.glb")
    assert {"walk", "static"} <= {a["name"] for a in cat.get("animations", [])}


# --------------------------------------------------------------------------- licences, notices, budget
def test_licences_and_third_party_notices():
    notices = (STATIC / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    for needle in ("three.js", "MIT", "Kenney", "KayKit", "CC0", "Cube Pets", "Mini Characters", "Furniture Bits"):
        assert needle in notices, needle
    for lic in ("assets/people/License.txt", "assets/pets/License.txt", "assets/furniture/LICENSE.txt"):
        text = (STATIC / lic).read_text(encoding="utf-8", errors="replace")
        assert "CC0" in text and "Creative Commons Zero" in text, lic


def test_static_dir_stays_under_15_mb():
    total = sum(f.stat().st_size for f in STATIC.rglob("*") if f.is_file())
    assert total < SIZE_BUDGET, f"static is {total / 1e6:.1f} MB"


# --------------------------------------------------------------------------- page scaffold + feed
def test_index_scaffold_for_3d_and_fallback():
    html = INDEX.read_text(encoding="utf-8")
    assert 'id="house3d"' in html and 'id="plan-wrap"' in html and 'id="plan"' in html
    assert 'id="view-3d"' in html and 'id="view-2d"' in html
    assert '<script type="module" src="static/3d/main.js"></script>' in html
    assert 'href="static/3d/scene.css"' in html
    # the 3D container starts hidden and the 2D plan starts checked: 2D is the fallback until WebGL2 reports in
    assert re.search(r'id="house3d"[^>]*\bhidden\b', html)
    assert re.search(r'id="view-2d"[^>]*aria-checked="true"', html)
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "window.homeFeed = feed" in app
    for ev in ("'state'", "'snapshot'", "'event'", "'settings'", "'motion'", "'view'", "'reset'"):
        assert f"feed.emit({ev}" in app, ev
    main = (STATIC / "3d" / "main.js").read_text(encoding="utf-8")
    assert "window.__scene3d" in main and "compileAsync" in main and "devicePixelRatio || 1, 2)" in main
    assert "feed.emit('scene3d'" in main and "WebGL2" in main


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_3d_modules_are_valid_es_modules():
    for f in SCENE_JS:
        proc = subprocess.run(["node", "--input-type=module", "--check"], input=f.read_bytes(), capture_output=True, timeout=30)
        assert proc.returncode == 0, f"{f.name}: {proc.stderr.decode(errors='replace')}"
    proc = subprocess.run(["node", "--check", str(STATIC / "app.js")], capture_output=True, timeout=30)
    assert proc.returncode == 0, proc.stderr.decode(errors="replace")


# --------------------------------------------------------------------------- served by the app
async def test_server_serves_vendor_and_assets():
    app = web.create_app(FakeRuntime(fast=True))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        r = await c.get("/static/vendor/three/build/three.module.js")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]
        r = await c.get("/static/3d/main.js")
        assert r.status_code == 200 and "javascript" in r.headers["content-type"]
        r = await c.get("/static/assets/people/character-male-a.glb")
        assert r.status_code == 200 and len(r.content) > 10_000
        r = await c.get("/static/assets/furniture/couch_pillows.gltf")
        assert r.status_code == 200 and json.loads(r.content)["asset"]
        r = await c.get("/static/assets/furniture/furniturebits_texture.png")
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert (await c.get("/static/assets/people/nope.glb")).status_code == 404
