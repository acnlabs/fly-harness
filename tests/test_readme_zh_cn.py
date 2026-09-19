"""Chinese README is a slim translation; English body stays English; version stays 0.5.5."""

from __future__ import annotations

import re
from pathlib import Path

from fly_harness import __version__

ROOT = Path(__file__).resolve().parents[1]
EN = ROOT / "README.md"
ZH = ROOT / "README.zh-CN.md"
FLYGYM_GIF = (
    "https://raw.githubusercontent.com/NeLy-EPFL/_media/main/flygym/overview_video.gif"
)
SWITCH = "[English](README.md) | [中文](README.zh-CN.md)"
CLIP = "docs/media/visual-demo/fly-walking.mp4"
CJK = re.compile(r"[\u4e00-\u9fff]")


def test_package_version_stays_0_5_5() -> None:
    assert __version__ == "0.5.5"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "0.5.5"' in pyproject
    assert 'readme = "README.md"' in pyproject


def test_language_switch_on_both_readmes() -> None:
    en = EN.read_text(encoding="utf-8")
    zh = ZH.read_text(encoding="utf-8")
    assert SWITCH in en.splitlines()[:6]
    assert SWITCH in zh.splitlines()[:6]
    assert en.startswith("# fly-harness\n")
    assert zh.startswith("# fly-harness\n")


def test_english_readme_body_stays_english_aside_from_switch() -> None:
    lines = EN.read_text(encoding="utf-8").splitlines()
    switch_lines = [line for line in lines if "中文" in line]
    assert switch_lines == [SWITCH]
    rest = "\n".join(line for line in lines if line != SWITCH)
    assert CJK.search(rest) is None
    assert CLIP in rest
    assert FLYGYM_GIF not in rest
    assert "**v0.5.5**" in rest


def test_zh_readme_leads_with_same_clip_and_short_caption() -> None:
    zh = ZH.read_text(encoding="utf-8")
    head = zh.split("## 安装", 1)[0]
    assert CLIP in head
    assert FLYGYM_GIF not in zh
    assert "overview_video.gif" not in zh
    assert "不是内核" in head
    assert "24 神经元 LIF fixture" in head
    assert "CPG" in head
    assert "FlyHarness.step" in head
    assert "**v0.5.5**" in head
    assert "24 神经元进程内 LIF fixture" in zh
    assert "不需要先有一台正在跑的仿真" in zh
    assert "替换 Model" in zh
    assert "examples/swap_brain.py" in zh
    assert "fly-harness-swap-brain-demo" in zh
    assert "你带来一台正在 tick" not in zh
    assert "公式" not in zh
    assert "Agent = Model + harness" not in zh
    assert head.index("NeuroMechFly") < head.index("**v0.5.5**")
    assert "## 用法导览" not in zh
    assert "### 1. Fixture" not in zh


def test_zh_readme_keeps_honest_non_claims_and_same_structure() -> None:
    en = EN.read_text(encoding="utf-8")
    zh = ZH.read_text(encoding="utf-8")
    assert "## 这不是什么" in zh
    assert "意识" in zh
    assert "上传" in zh
    assert "Google" in zh
    assert "166,700" in zh
    assert "docs/usage.md" in zh
    assert "docs/docking.md" in zh
    assert "fly-harness-check" in zh
    en_h2 = [line for line in en.splitlines() if line.startswith("## ")]
    zh_h2 = [line for line in zh.splitlines() if line.startswith("## ")]
    assert len(en_h2) == len(zh_h2) == 3
    assert "### 4." not in zh


def test_kernel_and_extras_untouched_by_zh_readme() -> None:
    src = ROOT / "src" / "fly_harness"
    for name in ("__init__.py", "harness.py", "backend.py"):
        text = (src / name).read_text(encoding="utf-8")
        assert "README.zh-CN.md" not in text
        assert "中文" not in text
    extras = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'readme = "README.md"' in extras
    assert 'version = "0.5.5"' in extras
