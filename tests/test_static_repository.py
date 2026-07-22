from pathlib import Path
def test_public_source_has_no_banned_legacy_terms_or_chinese_comments():
    root=Path(__file__).resolve().parents[1]; banned=["P"+"G-SCA","pg"+"_cnn","phys"+"_dualgate","phys"+"_recallgate","phys"+"_midfusion","phys"+"_early","physical"+"_fusion_mode","film"+"_scale","re"+"scue","hard"+"_max","Round"+" 8","Round"+" 11","Round"+" 15","Round"+" 16","Round"+" 19","final"+"9","final"+"_9","ziti"+".ttf","TIMES"+".TTF"]
    for folder in ("core","scripts","configs","tests"):
        for path in (root/folder).rglob("*.py"):
            text=path.read_text(encoding="utf-8"); assert not any(term.lower() in text.lower() for term in banned), path
