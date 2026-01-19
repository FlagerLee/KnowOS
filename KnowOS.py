from ConfigTree import Config
from LLM import ChatContext
from LLM_non_openai import ChatContext as CCReq
import os
import logging
import argparse


def parse_args():
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments including path, target, debug flag,
        output location, knowledge mode, and architecture settings.
    """
    parser = argparse.ArgumentParser()
    # positional argument: source tree path (project root containing Kconfig)
    parser.add_argument("path", default='.', type=str)
    # optional arguments
    parser.add_argument("-t", "--target", type=str)
    parser.add_argument("-d", "--debug", action="store_true")
    parser.add_argument("-o", "--output", default="config_output", type=str)
    parser.add_argument("-m", "--mode", default="hybrid", type=str)
    parser.add_argument("--use-knowledge", default=1, type=int)
    parser.add_argument("--cc", default="gcc", type=str)
    parser.add_argument("--ld", default="ld", type=str)
    parser.add_argument("--arch", default="x86", type=str)
    parser.add_argument("--srcarch", default="x86", type=str)
    parser.add_argument("--non-openai", action="store_true")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()
    return args


def main():
    """Main entrypoint: initialize environment, LLM context, then generate and save config.

    Steps:
    1. Parse command-line arguments.
    2. Configure logging verbosity for external libraries.
    3. Export environment variables used by downstream tools.
    4. Create a ChatContext (LLM wrapper) for interactive or automated prompts.
    5. Instantiate `Config` to parse Kconfig and produce final configuration.
    6. Run the configuration pipeline and save results.
    """
    args = parse_args()

    # propagate debug flag into global Config module
    DEBUG = args.debug

    # If knowledge usage is disabled, notify user (affects KG search behavior)
    if not bool(args.use_knowledge):
        print("Generating config without knowledge")

    # Reduce noisy logs from network/LLM libraries
    logging.getLogger("openai").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)

    # Export environment variables that other build tools or scripts may read
    os.environ["srctree"] = args.path
    os.environ["CC"] = args.cc
    os.environ["LD"] = args.ld
    os.environ["ARCH"] = args.arch
    os.environ["SRCARCH"] = args.srcarch

    # Initialize the LLM chat context. This object wraps model calls and pricing info.
    # Parameters: target (the configuration target), API key, API url and model name.
    if args.non_openai:
        chatter = CCReq(
            args.target, os.environ['NON_OPENAI_API_KEY'], os.environ['NON_OPENAI_URL'], model=args.model
        )
    else:
        chatter = ChatContext(
            args.target, os.environ['OPENAI_API_KEY'], os.environ['OPENAI_BASE_URL'], model="gpt-4o-mini"
        )

    # Build and run the configuration pipeline using the project's Kconfig file.
    # - first arg: path to Kconfig (e.g. /path/to/src/Kconfig)
    # - chatter: LLM interface used to ask/decide configuration items
    # - args.target: specific target to configure
    # - kg_search_mode: knowledge graph/search mode (e.g. 'hybrid')
    # - use_knowledge: whether to consult external knowledge sources
    # - config_path: where to read/write the generated .config file
    config = Config(
        f"{args.path}/Kconfig",
        chatter,
        args.target,
        kg_search_mode=args.mode,
        use_knowledge=bool(args.use_knowledge),
        config_path=f"{args.path}/.config",
    )
    # Execute the configuration generation process
    config.run()
    # Persist the generated configuration to the output location
    config.save(args.output)


if __name__ == "__main__":
    main()
