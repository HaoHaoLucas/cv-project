"""在 Windows 终端下强制使用 UTF-8 输出，解决中文乱码问题。

在任意脚本入口处 import 此模块即可：
    import src.utils.codec  # noqa
"""
import io
import sys


def _fix_stdio_encoding() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        if hasattr(stream, "buffer") and getattr(stream, "encoding", "").lower() != "utf-8":
            setattr(
                sys,
                stream_name,
                io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace"),
            )


_fix_stdio_encoding()
