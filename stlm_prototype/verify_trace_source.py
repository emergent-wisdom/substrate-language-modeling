"""Hash-check the frozen published source used by a derived TR-H run."""
import config
from write_run_manifest import verified_trace_source


def main():
    config.require_experiment_type("trace_readout")
    verified_trace_source()
    print(
        f"published source {config.SOURCE_RUN_ID!r} is intact; "
        "the derived-run protocol forbids writing to it")


if __name__ == "__main__":
    main()
