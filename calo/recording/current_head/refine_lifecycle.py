"""Tighten reviewed sequential tracking lifecycle in the isolated prepared clone; no build."""
from pathlib import Path
import argparse
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
REL = "Mu2eG4/inc/Mu2eG4BoundaryRecorder.hh"


def refine(root):
    manifest = json.loads((HERE / "source-manifest.json").read_text())
    target = root / REL
    original = target.read_text()
    if hashlib.sha256(original.encode()).hexdigest() != manifest["patched_sources"][REL]:
        raise ValueError("boundary lifecycle source drift")
    changes = [
        ("      bool saved = false;", "      bool saved = false;\n      bool running = false;\n      bool terminal = false;\n      G4TrackStatus lastEnd = fAlive;"),
        ('''      if (!inserted.second) {
        row("tracking_resume", trackFields(t, i) + ",\\\"state\\\":" + state(t));
        return;
      }
      i.saved = saved != nullptr;''', '''      if (!inserted.second) {
        if (i.running || i.terminal || (i.lastEnd != fSuspend && i.lastEnd != fSuspendAndWait))
          fail("invalid tracking resume lifecycle");
        const int resumedMembership = member(t.GetTouchable());
        if (resumedMembership < 0 || bool(resumedMembership) != i.active)
          fail("resume membership discontinuity or unavailable touchable");
        i.running = true;
        row("tracking_resume", trackFields(t, i) + ",\\\"state\\\":" + state(t));
        return;
      }
      if (t.GetCurrentStepNumber() != 0) fail("initial tracking start has nonzero step");
      i.running = true;
      i.saved = saved != nullptr;'''),
        ('''      auto& i = info(t);
      const auto& pre = *s.GetPreStepPoint();''', '''      auto& i = info(t);
      if (!i.running || i.terminal || t.GetCurrentStepNumber() < 1) fail("step outside running lifecycle");
      const auto& pre = *s.GetPreStepPoint();'''),
        ('''      if (i.birthUnknown) {
        i.birthUnknown = false;''', '''      if (i.birthUnknown) {
        if (t.GetCurrentStepNumber() != 1) fail("unresolved birth must appear at first step");
        i.birthUnknown = false;'''),
        ('''      auto& i = info(t);
      const bool terminal =''', '''      auto& i = info(t);
      if (!i.running || i.terminal) fail("tracking end outside running lifecycle");
      const bool terminal ='''),
        ('''      if (terminal) i.active = false;
    }''', '''      if (terminal) i.active = false;
      i.running = false;
      i.terminal = terminal;
      i.lastEnd = t.GetTrackStatus();
    }'''),
    ]
    new = original
    for old, replacement in changes:
        if new.count(old) != 1:
            raise ValueError("nonunique lifecycle anchor: " + old)
        new = new.replace(old, replacement)
    patch = "".join(difflib.unified_diff(original.splitlines(True), new.splitlines(True),
                                       fromfile="a/" + REL, tofile="b/" + REL))
    (HERE / "lifecycle-refinement.patch").write_text(patch)
    target.write_text(new)
    manifest["lifecycle_refinement"] = {"base_sha256": hashlib.sha256(original.encode()).hexdigest(),
                                         "patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}
    manifest["patched_sources"][REL] = hashlib.sha256(new.encode()).hexdigest()
    (HERE / "source-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", required=True, type=Path)
    refine(parser.parse_args().offline.resolve())
