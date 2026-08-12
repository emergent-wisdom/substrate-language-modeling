"""Hash-check the frozen published SPs used by a derived TR-C run."""

import config
from write_run_manifest import verified_contextual_source


def main():
    config.require_experiment_type("contextual_readout")
    source = verified_contextual_source()
    labels = ", ".join(sorted(source["checkpoints"]))
    print(
        f"published source {config.SOURCE_RUN_ID!r} is intact "
        f"({labels}); the derived-run protocol forbids writing to it")


if __name__ == "__main__":
    main()
