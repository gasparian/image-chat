"""Main entry point for the Gradio chat client."""

import sys
from .interface import create_interface
from .config import GRADIO_PORT


def main(server_port: int = GRADIO_PORT, share: bool = False):
    """Launch the Gradio chat interface.

    Args:
        server_port: Port for Gradio server
        share: Whether to create a public link
    """
    demo = create_interface()
    demo.launch(
        server_port=server_port,
        share=share,
        server_name="127.0.0.1"
    )


if __name__ == "__main__":
    share = "--share" in sys.argv
    main(share=share)
