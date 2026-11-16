"""Run every exact proof and write one record. Python standard library only.

    python proofs/run_all.py --output results/proofs.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import geometry  # noqa: E402
import partition  # noqa: E402
import residual  # noqa: E402
import separation  # noqa: E402
import storage  # noqa: E402

PROOFS = {"terminal_jacobian_separation": separation, "omitted_influence_residual": residual,
          "storage": storage, "partition": partition, "response_geometry": geometry}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    record = {"status": "passed", "proofs": {}}
    for name, module in PROOFS.items():
        started = time.process_time()
        result = module.check()
        record["proofs"][name] = {"result": result, "cpu_seconds": time.process_time() - started,
                                  "source_sha256": hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()}
    record["shared_algebra_sha256"] = hashlib.sha256((HERE / "_exact.py").read_bytes()).hexdigest()
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:  # never overwrite evidence
            stream.write(text)
    print("all exact proofs passed:", ", ".join(PROOFS))


if __name__ == "__main__":
    main()
